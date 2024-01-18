//===- CCAS.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-c/CAS.h"

#include "CASUtils.h"
#include "CXError.h"
#include "CXString.h"

#include "clang/Basic/LLVM.h"
#include "clang/CAS/CASOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompileJobCache.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::cas;
using llvm::Error;

namespace {

struct WrappedCASObject {
  ObjectProxy Obj;
  std::shared_ptr<llvm::cas::ObjectStore> CAS;
};

struct WrappedCachedCompilation {
  CASID CacheKey;
  clang::cas::CompileJobCacheResult CachedResult;
  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> AC;

  static CXCASCachedCompilation
  fromResultID(Expected<std::optional<CASID>> ResultID, CASID CacheKey,
               const std::shared_ptr<llvm::cas::ObjectStore> &CAS,
               const std::shared_ptr<llvm::cas::ActionCache> &AC,
               CXError *OutError);
};

struct WrappedReplayResult {
  SmallString<256> DiagText;
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedCASObject, CXCASObject)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedCachedCompilation,
                                   CXCASCachedCompilation)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedReplayResult, CXCASReplayResult)

} // anonymous namespace

static void passAsCXError(Error &&E, CXError *OutError) {
  if (OutError)
    *OutError = cxerror::create(std::move(E));
  else
    llvm::consumeError(std::move(E));
}

CXCASCachedCompilation WrappedCachedCompilation::fromResultID(
    Expected<std::optional<CASID>> ResultID, CASID CacheKey,
    const std::shared_ptr<llvm::cas::ObjectStore> &CAS,
    const std::shared_ptr<llvm::cas::ActionCache> &AC, CXError *OutError) {

  auto failure = [OutError](Error &&E) -> CXCASCachedCompilation {
    passAsCXError(std::move(E), OutError);
    return nullptr;
  };

  if (!ResultID)
    return failure(ResultID.takeError());
  if (!*ResultID)
    return nullptr;

  auto OptResultRef = CAS->getReference(**ResultID);
  if (!OptResultRef)
    return nullptr;

  clang::cas::CompileJobResultSchema Schema(*CAS);
  auto CachedResult = Schema.load(*OptResultRef);
  if (!CachedResult)
    return failure(CachedResult.takeError());
  return wrap(new WrappedCachedCompilation{std::move(CacheKey),
                                           std::move(*CachedResult), CAS, AC});
}

CXCASOptions clang_experimental_cas_Options_create(void) {
  return wrap(new CASOptions());
}

void clang_experimental_cas_Options_dispose(CXCASOptions Opts) {
  delete unwrap(Opts);
}

void clang_experimental_cas_Options_setOnDiskPath(CXCASOptions COpts,
                                                  const char *Path) {
  CASOptions &Opts = *unwrap(COpts);
  Opts.CASPath = Path;
}

void clang_experimental_cas_Options_setPluginPath(CXCASOptions COpts,
                                                  const char *Path) {
  CASOptions &Opts = *unwrap(COpts);
  Opts.PluginPath = Path;
}

void clang_experimental_cas_Options_setPluginOption(CXCASOptions COpts,
                                                    const char *Name,
                                                    const char *Value) {
  CASOptions &Opts = *unwrap(COpts);
  Opts.PluginOptions.emplace_back(Name, Value);
}

CXCASDatabases clang_experimental_cas_Databases_create(CXCASOptions COpts,
                                                       CXString *Error) {
  CASOptions &Opts = *unwrap(COpts);

  SmallString<128> DiagBuf;
  llvm::raw_svector_ostream OS(DiagBuf);
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagPrinter(OS, DiagOpts.get());
  DiagnosticsEngine Diags(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), DiagOpts.get(),
      &DiagPrinter, /*ShouldOwnClient=*/false);

  auto [CAS, Cache] = Opts.getOrCreateDatabases(Diags);
  if (!CAS || !Cache) {
    if (Error)
      *Error = cxstring::createDup(OS.str());
    return nullptr;
  }

  return wrap(new WrappedCASDatabases{Opts, std::move(CAS), std::move(Cache)});
}

void clang_experimental_cas_Databases_dispose(CXCASDatabases CDBs) {
  delete unwrap(CDBs);
}

int64_t clang_experimental_cas_Databases_get_storage_size(CXCASDatabases CDBs,
                                                          CXError *OutError) {
  // Commonly used ObjectStore implementations (on-disk and plugin) combine a
  // CAS and action-cache into a single directory managing the storage
  // holistically for both, so calling the ObjectStore API is sufficient.
  // FIXME: For completeness we should figure out how to deal with potential
  // implementations that use separate directories for CAS and action-cache.
  std::optional<uint64_t> Size;
  if (Error E = unwrap(CDBs)->CAS->getStorageSize().moveInto(Size)) {
    passAsCXError(std::move(E), OutError);
    return -2;
  }
  if (!Size)
    return -1;
  return *Size;
}

CXError clang_experimental_cas_Databases_set_size_limit(CXCASDatabases CDBs,
                                                        int64_t size_limit) {
  // Commonly used ObjectStore implementations (on-disk and plugin) combine a
  // CAS and action-cache into a single directory managing the storage
  // holistically for both, so calling the ObjectStore API is sufficient.
  // FIXME: For completeness we should figure out how to deal with potential
  // implementations that use separate directories for CAS and action-cache.
  std::optional<uint64_t> SizeLimit;
  if (size_limit < 0) {
    return cxerror::create(llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid size limit passed to "
        "clang_experimental_cas_Databases_set_size_limit"));
  }
  if (size_limit > 0) {
    SizeLimit = size_limit;
  }
  if (Error E = unwrap(CDBs)->CAS->setSizeLimit(SizeLimit))
    return cxerror::create(std::move(E));
  return nullptr;
}

CXError
clang_experimental_cas_Databases_prune_ondisk_data(CXCASDatabases CDBs) {
  // Commonly used ObjectStore implementations (on-disk and plugin) combine a
  // CAS and action-cache into a single directory managing the storage
  // holistically for both, so calling the ObjectStore API is sufficient.
  // FIXME: For completeness we should figure out how to deal with potential
  // implementations that use separate directories for CAS and action-cache.
  if (Error E = unwrap(CDBs)->CAS->pruneStorageData())
    return cxerror::create(std::move(E));
  return nullptr;
}

CXCASObject clang_experimental_cas_loadObjectByString(CXCASDatabases CDBs,
                                                      const char *PrintedID,
                                                      CXError *OutError) {
  WrappedCASDatabases &DBs = *unwrap(CDBs);
  ObjectStore &CAS = *DBs.CAS;

  if (OutError)
    *OutError = nullptr;

  auto failure = [OutError](Error &&E) -> CXCASObject {
    passAsCXError(std::move(E), OutError);
    return nullptr;
  };

  Expected<CASID> Digest = CAS.parseID(PrintedID);
  if (!Digest)
    return failure(Digest.takeError());
  std::optional<ObjectRef> Ref = CAS.getReference(*Digest);
  if (!Ref)
    return nullptr;

  // Visit the graph of the object to ensure it's fully materialized.

  SmallVector<ObjectRef> ObjectsToLoad;
  ObjectsToLoad.push_back(*Ref);
  llvm::SmallDenseSet<ObjectRef> ObjectsSeen;

  while (!ObjectsToLoad.empty()) {
    ObjectRef Ref = ObjectsToLoad.pop_back_val();
    bool Inserted = ObjectsSeen.insert(Ref).second;
    if (!Inserted)
      continue;
    std::optional<ObjectProxy> Obj;
    if (Error E = CAS.getProxy(Ref).moveInto(Obj))
      return failure(std::move(E));
    if (Error E = Obj->forEachReference([&ObjectsToLoad](ObjectRef R) -> Error {
          ObjectsToLoad.push_back(R);
          return Error::success();
        }))
      return failure(std::move(E));
  }

  std::optional<ObjectProxy> Obj;
  if (Error E = CAS.getProxy(*Ref).moveInto(Obj))
    return failure(std::move(E));

  if (!Obj)
    return nullptr;
  return wrap(new WrappedCASObject{std::move(*Obj), DBs.CAS});
}

void clang_experimental_cas_loadObjectByString_async(
    CXCASDatabases CDBs, const char *PrintedID, void *Ctx,
    void (*Callback)(void *Ctx, CXCASObject, CXError),
    CXCASCancellationToken *OutToken) {
  if (OutToken)
    *OutToken = nullptr;
  WrappedCASDatabases &DBs = *unwrap(CDBs);
  ObjectStore &CAS = *DBs.CAS;

  Expected<CASID> Digest = CAS.parseID(PrintedID);
  if (!Digest)
    return Callback(Ctx, nullptr, cxerror::create(Digest.takeError()));
  std::optional<ObjectRef> Ref = CAS.getReference(*Digest);
  if (!Ref)
    return Callback(Ctx, nullptr, nullptr);

  /// Asynchronously visits the graph of the object node to ensure it's fully
  /// materialized.
  class AsyncObjectLoader
      : public std::enable_shared_from_this<AsyncObjectLoader> {
    void *Ctx;
    void (*Callback)(void *Ctx, CXCASObject, CXError);
    std::shared_ptr<cas::ObjectStore> CAS;

    llvm::SmallDenseSet<ObjectRef> ObjectsSeen;
    unsigned NumPending = 0;
    std::optional<ObjectProxy> RootObj;
    std::atomic<bool> MissingNode{false};
    /// The first error that occurred.
    std::optional<Error> ErrOccurred;
    std::mutex Mutex;

  public:
    AsyncObjectLoader(void *Ctx,
                      void (*Callback)(void *Ctx, CXCASObject, CXError),
                      std::shared_ptr<cas::ObjectStore> CAS)
        : Ctx(Ctx), Callback(Callback), CAS(std::move(CAS)) {}

    void visit(ObjectRef Ref, bool IsRootNode) {
      bool Inserted;
      {
        std::lock_guard<std::mutex> Guard(Mutex);
        Inserted = ObjectsSeen.insert(Ref).second;
        if (Inserted)
          ++NumPending;
      }
      if (!Inserted) {
        finishedNode();
        return;
      }
      auto This = shared_from_this();
      CAS->getProxyAsync(
          Ref, [This, IsRootNode](Expected<std::optional<ObjectProxy>> Obj) {
            auto _1 = llvm::make_scope_exit([&]() { This->finishedNode(); });
            if (!Obj) {
              This->encounteredError(Obj.takeError());
              return;
            }
            if (!*Obj) {
              This->MissingNode = true;
              return;
            }
            if (IsRootNode)
              This->RootObj = *Obj;
            cantFail((*Obj)->forEachReference([&This](ObjectRef Sub) -> Error {
              This->visit(Sub, /*IsRootNode*/ false);
              return Error::success();
            }));
          });
    }

    void finishedNode() {
      bool FinishedPending;
      {
        std::lock_guard<std::mutex> Guard(Mutex);
        assert(NumPending);
        --NumPending;
        FinishedPending = (NumPending == 0);
      }
      if (!FinishedPending)
        return;

      if (ErrOccurred)
        return Callback(Ctx, nullptr, cxerror::create(std::move(*ErrOccurred)));
      if (MissingNode)
        return Callback(Ctx, nullptr, nullptr);
      return Callback(
          Ctx, wrap(new WrappedCASObject{std::move(*RootObj), std::move(CAS)}),
          nullptr);
    }

    /// Only keeps the first error that occurred.
    void encounteredError(Error &&E) {
      std::lock_guard<std::mutex> Guard(Mutex);
      if (ErrOccurred) {
        llvm::consumeError(std::move(E));
        return;
      }
      ErrOccurred = std::move(E);
    }
  };

  auto WL = std::make_shared<AsyncObjectLoader>(Ctx, Callback, DBs.CAS);
  WL->visit(*Ref, /*IsRootNode*/ true);
}

void clang_experimental_cas_CASObject_dispose(CXCASObject CObj) {
  delete unwrap(CObj);
}

CXCASCachedCompilation
clang_experimental_cas_getCachedCompilation(CXCASDatabases CDBs,
                                            const char *CacheKey, bool Globally,
                                            CXError *OutError) {
  WrappedCASDatabases &DBs = *unwrap(CDBs);

  if (OutError)
    *OutError = nullptr;

  auto failure = [OutError](Error &&E) -> CXCASCachedCompilation {
    passAsCXError(std::move(E), OutError);
    return nullptr;
  };

  Expected<CASID> KeyID = DBs.CAS->parseID(CacheKey);
  if (!KeyID)
    return failure(KeyID.takeError());

  return WrappedCachedCompilation::fromResultID(
      DBs.Cache->get(*KeyID, Globally), *KeyID, DBs.CAS, DBs.Cache, OutError);
}

void clang_experimental_cas_getCachedCompilation_async(
    CXCASDatabases CDBs, const char *CacheKey, bool Globally, void *Ctx,
    void (*Callback)(void *Ctx, CXCASCachedCompilation, CXError),
    CXCASCancellationToken *OutToken) {
  if (OutToken)
    *OutToken = nullptr;
  WrappedCASDatabases &DBs = *unwrap(CDBs);

  Expected<CASID> KeyID = DBs.CAS->parseID(CacheKey);
  if (!KeyID)
    return Callback(Ctx, nullptr, cxerror::create(KeyID.takeError()));

  DBs.Cache->getAsync(*KeyID, Globally,
                      [KeyID = *KeyID, CAS = DBs.CAS, AC = DBs.Cache, Ctx,
                       Callback](Expected<std::optional<CASID>> ResultID) {
                        CXError Err = nullptr;
                        CXCASCachedCompilation CComp =
                            WrappedCachedCompilation::fromResultID(
                                std::move(ResultID), std::move(KeyID),
                                std::move(CAS), std::move(AC), &Err);
                        Callback(Ctx, CComp, Err);
                      });
}

void clang_experimental_cas_CachedCompilation_dispose(
    CXCASCachedCompilation CComp) {
  delete unwrap(CComp);
}

size_t clang_experimental_cas_CachedCompilation_getNumOutputs(
    CXCASCachedCompilation CComp) {
  return unwrap(CComp)->CachedResult.getNumOutputs();
}

CXString clang_experimental_cas_CachedCompilation_getOutputName(
    CXCASCachedCompilation CComp, size_t OutputIdx) {
  CompileJobCacheResult::Output Output =
      unwrap(CComp)->CachedResult.getOutput(OutputIdx);
  return cxstring::createRef(
      CompileJobCacheResult::getOutputKindName(Output.Kind));
}

CXString clang_experimental_cas_CachedCompilation_getOutputCASIDString(
    CXCASCachedCompilation CComp, size_t OutputIdx) {
  WrappedCachedCompilation &WComp = *unwrap(CComp);
  CompileJobCacheResult::Output Output =
      WComp.CachedResult.getOutput(OutputIdx);
  return cxstring::createDup(WComp.CAS->getID(Output.Object).toString());
}

bool clang_experimental_cas_CachedCompilation_isOutputMaterialized(
    CXCASCachedCompilation CComp, size_t OutputIdx) {
  WrappedCachedCompilation &WComp = *unwrap(CComp);
  CompileJobCacheResult::Output Output =
      WComp.CachedResult.getOutput(OutputIdx);
  bool IsMaterialized = false;
  // FIXME: Propagate error to caller instead of calling `report_fatal_error`.
  // In practice this should not fail because it checks the local CAS only.
  if (Error E =
          WComp.CAS->isMaterialized(Output.Object).moveInto(IsMaterialized))
    llvm::report_fatal_error(std::move(E));
  return IsMaterialized;
}

void clang_experimental_cas_CachedCompilation_makeGlobal(
    CXCASCachedCompilation CComp, void *Ctx,
    void (*Callback)(void *Ctx, CXError), CXCASCancellationToken *OutToken) {
  if (OutToken)
    *OutToken = nullptr;
  WrappedCachedCompilation &WComp = *unwrap(CComp);
  CompileJobCacheResult &CacheResult = WComp.CachedResult;
  WComp.AC->putAsync(WComp.CacheKey, CacheResult.getID(), /*Globally=*/true,
                     [Ctx, Callback](Error E) {
                       Callback(Ctx, cxerror::create(std::move(E)));
                     });
}

CXCASReplayResult clang_experimental_cas_replayCompilation(
    CXCASCachedCompilation CComp, int argc, const char *const *argv,
    const char *WorkingDirectory, void * /*reserved*/, CXError *OutError) {
  WrappedCachedCompilation &WComp = *unwrap(CComp);

  if (OutError)
    *OutError = nullptr;

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  SmallString<128> DiagsBuffer;
  llvm::raw_svector_ostream DiagOS(DiagsBuffer);
  auto *DiagsPrinter = new TextDiagnosticPrinter(DiagOS, DiagOpts.get());
  DiagnosticsEngine Diags(DiagID, DiagOpts.get(), DiagsPrinter);

  SmallVector<const char *, 256> Args(argv, argv + argc);
  llvm::BumpPtrAllocator Alloc;
  if (llvm::Error E = driver::expandResponseFiles(Args, /*CLMode=*/false, Alloc)) {
    Diags.Report(diag::err_drv_expand_response_file)
        << llvm::toString(std::move(E));
    if (OutError)
      *OutError = cxerror::create(DiagOS.str());
    return nullptr;
  }

  auto Invok = std::make_shared<CompilerInvocation>();
  bool Success = CompilerInvocation::CreateFromArgs(*Invok, ArrayRef(Args).drop_front(),
                                                    Diags, Args.front());
  if (!Success) {
    if (OutError)
      *OutError = cxerror::create(DiagOS.str());
    return nullptr;
  }

  SmallString<256> DiagText;
  std::optional<int> Ret;
  if (Error E = CompileJobCache::replayCachedResult(
                    std::move(Invok), WorkingDirectory, WComp.CacheKey,
                    WComp.CachedResult, DiagText)
                    .moveInto(Ret)) {
    passAsCXError(std::move(E), OutError);
    return nullptr;
  }

  if (!Ret)
    return nullptr;
  // If there was no CAS error and the compilation was cached it will be
  // 'success', we don't cache compilation failures.
  assert(*Ret == 0);
  return wrap(new WrappedReplayResult{std::move(DiagText)});
}

void clang_experimental_cas_ReplayResult_dispose(CXCASReplayResult CRR) {
  delete unwrap(CRR);
}

CXString clang_experimental_cas_ReplayResult_getStderr(CXCASReplayResult CRR) {
  return cxstring::createDup(unwrap(CRR)->DiagText);
}

void clang_experimental_cas_CancellationToken_cancel(CXCASCancellationToken) {
  // FIXME: Implement.
}

void clang_experimental_cas_CancellationToken_dispose(CXCASCancellationToken) {
  // FIXME: Implement.
}

void clang_experimental_cas_ObjectStore_dispose(CXCASObjectStore CAS) {
  delete unwrap(CAS);
}
void clang_experimental_cas_ActionCache_dispose(CXCASActionCache Cache) {
  delete unwrap(Cache);
}

CXCASObjectStore
clang_experimental_cas_OnDiskObjectStore_create(const char *Path,
                                                CXString *Error) {
  auto CAS = llvm::cas::createOnDiskCAS(Path);
  if (!CAS) {
    if (Error)
      *Error = cxstring::createDup(llvm::toString(CAS.takeError()));
    return nullptr;
  }
  return wrap(new WrappedObjectStore{std::move(*CAS), Path});
}

CXCASActionCache
clang_experimental_cas_OnDiskActionCache_create(const char *Path,
                                                CXString *Error) {
  auto Cache = llvm::cas::createOnDiskActionCache(Path);
  if (!Cache) {
    if (Error)
      *Error = cxstring::createDup(llvm::toString(Cache.takeError()));
    return nullptr;
  }
  return wrap(new WrappedActionCache{std::move(*Cache), Path});
}
