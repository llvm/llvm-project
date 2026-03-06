//===- ModuleManager.cpp - Module Manager ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleManager class, which manages a set of loaded
//  modules for the ASTReader.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ModuleManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Serialization/GlobalModuleIndex.h"
#include "clang/Serialization/InMemoryModuleCache.h"
#include "clang/Serialization/ModuleCache.h"
#include "clang/Serialization/ModuleFile.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <cassert>
#include <memory>
#include <string>
#include <system_error>

using namespace clang;
using namespace serialization;

ModuleFile *ModuleManager::lookupByFileName(StringRef Name) const {
  SmallString<128> NormalizedFileName = Name;
  llvm::sys::fs::make_absolute(NormalizedFileName);
  llvm::sys::path::make_preferred(NormalizedFileName);
  return Modules.lookup(NormalizedFileName);
}

ModuleFile *ModuleManager::lookupByModuleName(StringRef Name) const {
  if (const Module *Mod = HeaderSearchInfo.getModuleMap().findModule(Name))
    if (OptionalFileEntryRef File = Mod->getASTFile())
      return lookup(*File);

  return nullptr;
}

ModuleFile *ModuleManager::lookup(FileEntryRef File) const {
  llvm::SmallString<128> NormalizedFileName = File.getName();
  llvm::sys::fs::make_absolute(NormalizedFileName);
  llvm::sys::path::make_preferred(NormalizedFileName);
  return Modules.lookup(NormalizedFileName);
}

std::unique_ptr<llvm::MemoryBuffer>
ModuleManager::lookupBuffer(StringRef Name) {
  return std::move(InMemoryBuffers[Name]);
}

static bool checkModuleFile(const FileEntry *File, off_t ExpectedSize,
                            time_t ExpectedModTime, std::string &ErrorStr) {
  assert(File && "Checking expectations of a non-existent module file");

  if (ExpectedSize && ExpectedSize != File->getSize()) {
    ErrorStr = "module file has a different size than expected";
    return true;
  }

  if (ExpectedModTime && ExpectedModTime != File->getModificationTime()) {
    ErrorStr = "module file has a different modification time than expected";
    return true;
  }

  return false;
}

static bool checkSignature(ASTFileSignature Signature,
                           ASTFileSignature ExpectedSignature,
                           std::string &ErrorStr) {
  if (!ExpectedSignature || Signature == ExpectedSignature)
    return false;

  ErrorStr =
      Signature ? "signature mismatch" : "could not read module signature";
  return true;
}

static void updateModuleImports(ModuleFile &MF, ModuleFile *ImportedBy,
                                SourceLocation ImportLoc) {
  if (ImportedBy) {
    MF.ImportedBy.insert(ImportedBy);
    ImportedBy->Imports.insert(&MF);
  } else {
    if (!MF.DirectlyImported)
      MF.ImportLoc = ImportLoc;

    MF.DirectlyImported = true;
  }
}

ModuleManager::AddModuleResult
ModuleManager::addModule(StringRef FileName, ModuleKind Type,
                         SourceLocation ImportLoc, ModuleFile *ImportedBy,
                         unsigned Generation,
                         off_t ExpectedSize, time_t ExpectedModTime,
                         ASTFileSignature ExpectedSignature,
                         ASTFileSignatureReader ReadSignature,
                         ModuleFile *&Module,
                         std::string &ErrorStr) {
  Module = nullptr;

  // FIXME: Only call this after we figure out we have not loaded this module
  // before to avoid unnecessary IO.
  uint64_t InputFilesValidationTimestamp = 0;
  if (Type == MK_ImplicitModule)
    InputFilesValidationTimestamp = ModCache.getModuleTimestamp(FileName);

  SmallString<128> NormalizedFileName = FileName;
  llvm::sys::fs::make_absolute(NormalizedFileName);
  llvm::sys::path::make_preferred(NormalizedFileName);

  bool IgnoreModTime = Type == MK_ExplicitModule || Type == MK_PrebuiltModule;
  if (ImportedBy)
    IgnoreModTime &= ImportedBy->Kind == MK_ExplicitModule ||
                     ImportedBy->Kind == MK_PrebuiltModule;
  if (IgnoreModTime) {
    // If neither this file nor the importer are in the module cache, this file
    // might have a different mtime due to being moved across filesystems in
    // a distributed build. The size must still match, though. (As must the
    // contents, but we can't check that.)
    ExpectedModTime = 0;
  }

  // Check whether we already loaded this module, before
  if (ModuleFile *ModuleEntry = Modules.lookup(NormalizedFileName)) {
    // Check file properties.
    if (checkModuleFile(ModuleEntry->File, ExpectedSize, ExpectedModTime,
                        ErrorStr))
      return OutOfDate;

    // Check the stored signature.
    if (checkSignature(ModuleEntry->Signature, ExpectedSignature, ErrorStr))
      return OutOfDate;

    Module = ModuleEntry;
    updateModuleImports(*ModuleEntry, ImportedBy, ImportLoc);
    return AlreadyLoaded;
  }

  // Load the contents of the module
  OptionalFileEntryRef Entry;
  llvm::MemoryBuffer *ModuleBuffer = nullptr;
  std::unique_ptr<llvm::MemoryBuffer> NewFileBuffer = nullptr;
  if (std::unique_ptr<llvm::MemoryBuffer> Buffer =
          lookupBuffer(NormalizedFileName)) {
    // The buffer was already provided for us.
    ModuleBuffer = &getModuleCache().getInMemoryModuleCache().addBuiltPCM(
        FileName, std::move(Buffer));
  } else if (llvm::MemoryBuffer *Buffer =
                 getModuleCache().getInMemoryModuleCache().lookupPCM(
                     FileName)) {
    ModuleBuffer = Buffer;
  } else if (getModuleCache().getInMemoryModuleCache().shouldBuildPCM(
                 FileName)) {
    // Report that the module is out of date, since we tried (and failed) to
    // import it earlier.
    return OutOfDate;
  } else {
    Entry = FileName == "-"
                ? expectedToOptional(FileMgr.getSTDIN())
                : FileMgr.getOptionalFileRef(FileName, /*OpenFile=*/true,
                                             /*CacheFailure=*/false);
    if (!Entry) {
      ErrorStr = "module file not found";
      return Missing;
    }

    // FIXME: Consider moving this after this else branch so that we check
    // size/mtime expectations even when pulling the module file out of the
    // in-memory module cache or the provided in-memory buffers.
    // Check file properties.
    if (checkModuleFile(*Entry, ExpectedSize, ExpectedModTime, ErrorStr))
      return OutOfDate;

    // Get a buffer of the file and close the file descriptor when done.
    // The file is volatile because in a parallel build we expect multiple
    // compiler processes to use the same module file rebuilding it if needed.
    //
    // RequiresNullTerminator is false because module files don't need it, and
    // this allows the file to still be mmapped.
    auto Buf = FileMgr.getBufferForFile(*Entry,
                                        /*IsVolatile=*/true,
                                        /*RequiresNullTerminator=*/false);

    if (!Buf) {
      ErrorStr = Buf.getError().message();
      return Missing;
    }

    NewFileBuffer = std::move(*Buf);
    ModuleBuffer = NewFileBuffer.get();
  }

  if (!Entry) {
    // Unless we loaded the buffer from a freshly open file (else branch above),
    // we don't have any FileEntry for this ModuleFile. Make one up.
    // FIXME: Make it so that ModuleFile is not tied to a FileEntry.
    Entry = FileMgr.getVirtualFileRef(FileName, ExpectedSize, ExpectedModTime);
  }

  // Allocate a new module.
  auto NewModule = std::make_unique<ModuleFile>(Type, *Entry, Generation);
  NewModule->Index = Chain.size();
  NewModule->FileName = FileName.str();
  NewModule->ImportLoc = ImportLoc;
  NewModule->InputFilesValidationTimestamp = InputFilesValidationTimestamp;
  NewModule->Buffer = ModuleBuffer;
  // Initialize the stream.
  NewModule->Data = PCHContainerRdr.ExtractPCH(*NewModule->Buffer);

  // Read the signature eagerly now so that we can check it.  Avoid calling
  // ReadSignature unless there's something to check though.
  if (ExpectedSignature && checkSignature(ReadSignature(NewModule->Data),
                                          ExpectedSignature, ErrorStr))
    return OutOfDate;

  if (NewFileBuffer)
    getModuleCache().getInMemoryModuleCache().addPCM(FileName,
                                                     std::move(NewFileBuffer));

  // We're keeping this module.  Store it everywhere.
  Module = Modules[NormalizedFileName] = NewModule.get();

  updateModuleImports(*NewModule, ImportedBy, ImportLoc);

  if (!NewModule->isModule())
    PCHChain.push_back(NewModule.get());
  if (!ImportedBy)
    Roots.push_back(NewModule.get());

  Chain.push_back(std::move(NewModule));
  return NewlyLoaded;
}

void ModuleManager::removeModules(ModuleIterator First) {
  auto Last = end();
  if (First == Last)
    return;

  // Explicitly clear VisitOrder since we might not notice it is stale.
  VisitOrder.clear();

  // Collect the set of module file pointers that we'll be removing.
  llvm::SmallPtrSet<ModuleFile *, 4> victimSet(
      (llvm::pointer_iterator<ModuleIterator>(First)),
      (llvm::pointer_iterator<ModuleIterator>(Last)));

  auto IsVictim = [&](ModuleFile *MF) {
    return victimSet.count(MF);
  };
  // Remove any references to the now-destroyed modules.
  for (auto I = begin(); I != First; ++I) {
    I->Imports.remove_if(IsVictim);
    I->ImportedBy.remove_if(IsVictim);
  }
  llvm::erase_if(Roots, IsVictim);

  // Remove the modules from the PCH chain.
  for (auto I = First; I != Last; ++I) {
    if (!I->isModule()) {
      PCHChain.erase(llvm::find(PCHChain, &*I), PCHChain.end());
      break;
    }
  }

  // Delete the modules.
  for (ModuleIterator victim = First; victim != Last; ++victim)
    Modules.erase(victim->File.getName());

  Chain.erase(Chain.begin() + (First - begin()), Chain.end());
}

void
ModuleManager::addInMemoryBuffer(StringRef FileName,
                                 std::unique_ptr<llvm::MemoryBuffer> Buffer) {
  SmallString<128> NormalizedFileName = FileName;
  llvm::sys::fs::make_absolute(NormalizedFileName);
  llvm::sys::path::make_preferred(NormalizedFileName);
  InMemoryBuffers[NormalizedFileName] = std::move(Buffer);
}

std::unique_ptr<ModuleManager::VisitState> ModuleManager::allocateVisitState() {
  // Fast path: if we have a cached state, use it.
  if (FirstVisitState) {
    auto Result = std::move(FirstVisitState);
    FirstVisitState = std::move(Result->NextState);
    return Result;
  }

  // Allocate and return a new state.
  return std::make_unique<VisitState>(size());
}

void ModuleManager::returnVisitState(std::unique_ptr<VisitState> State) {
  assert(State->NextState == nullptr && "Visited state is in list?");
  State->NextState = std::move(FirstVisitState);
  FirstVisitState = std::move(State);
}

void ModuleManager::setGlobalIndex(GlobalModuleIndex *Index) {
  GlobalIndex = Index;
  if (!GlobalIndex) {
    ModulesInCommonWithGlobalIndex.clear();
    return;
  }

  // Notify the global module index about all of the modules we've already
  // loaded.
  for (ModuleFile &M : *this)
    if (!GlobalIndex->loadedModuleFile(&M))
      ModulesInCommonWithGlobalIndex.push_back(&M);
}

void ModuleManager::moduleFileAccepted(ModuleFile *MF) {
  if (!GlobalIndex || GlobalIndex->loadedModuleFile(MF))
    return;

  ModulesInCommonWithGlobalIndex.push_back(MF);
}

ModuleManager::ModuleManager(FileManager &FileMgr, ModuleCache &ModCache,
                             const PCHContainerReader &PCHContainerRdr,
                             const HeaderSearch &HeaderSearchInfo)
    : FileMgr(FileMgr), ModCache(ModCache), PCHContainerRdr(PCHContainerRdr),
      HeaderSearchInfo(HeaderSearchInfo) {}

void ModuleManager::visit(llvm::function_ref<bool(ModuleFile &M)> Visitor,
                          llvm::SmallPtrSetImpl<ModuleFile *> *ModuleFilesHit) {
  // If the visitation order vector is the wrong size, recompute the order.
  if (VisitOrder.size() != Chain.size()) {
    unsigned N = size();
    VisitOrder.clear();
    VisitOrder.reserve(N);

    // Record the number of incoming edges for each module. When we
    // encounter a module with no incoming edges, push it into the queue
    // to seed the queue.
    SmallVector<ModuleFile *, 4> Queue;
    Queue.reserve(N);
    llvm::SmallVector<unsigned, 4> UnusedIncomingEdges;
    UnusedIncomingEdges.resize(size());
    for (ModuleFile &M : llvm::reverse(*this)) {
      unsigned Size = M.ImportedBy.size();
      UnusedIncomingEdges[M.Index] = Size;
      if (!Size)
        Queue.push_back(&M);
    }

    // Traverse the graph, making sure to visit a module before visiting any
    // of its dependencies.
    while (!Queue.empty()) {
      ModuleFile *CurrentModule = Queue.pop_back_val();
      VisitOrder.push_back(CurrentModule);

      // For any module that this module depends on, push it on the
      // stack (if it hasn't already been marked as visited).
      for (ModuleFile *M : llvm::reverse(CurrentModule->Imports)) {
        // Remove our current module as an impediment to visiting the
        // module we depend on. If we were the last unvisited module
        // that depends on this particular module, push it into the
        // queue to be visited.
        unsigned &NumUnusedEdges = UnusedIncomingEdges[M->Index];
        if (NumUnusedEdges && (--NumUnusedEdges == 0))
          Queue.push_back(M);
      }
    }

    assert(VisitOrder.size() == N && "Visitation order is wrong?");

    FirstVisitState = nullptr;
  }

  auto State = allocateVisitState();
  unsigned VisitNumber = State->NextVisitNumber++;

  // If the caller has provided us with a hit-set that came from the global
  // module index, mark every module file in common with the global module
  // index that is *not* in that set as 'visited'.
  if (ModuleFilesHit && !ModulesInCommonWithGlobalIndex.empty()) {
    for (unsigned I = 0, N = ModulesInCommonWithGlobalIndex.size(); I != N; ++I)
    {
      ModuleFile *M = ModulesInCommonWithGlobalIndex[I];
      if (!ModuleFilesHit->count(M))
        State->VisitNumber[M->Index] = VisitNumber;
    }
  }

  for (unsigned I = 0, N = VisitOrder.size(); I != N; ++I) {
    ModuleFile *CurrentModule = VisitOrder[I];
    // Should we skip this module file?
    if (State->VisitNumber[CurrentModule->Index] == VisitNumber)
      continue;

    // Visit the module.
    assert(State->VisitNumber[CurrentModule->Index] == VisitNumber - 1);
    State->VisitNumber[CurrentModule->Index] = VisitNumber;
    if (!Visitor(*CurrentModule))
      continue;

    // The visitor has requested that cut off visitation of any
    // module that the current module depends on. To indicate this
    // behavior, we mark all of the reachable modules as having been visited.
    ModuleFile *NextModule = CurrentModule;
    do {
      // For any module that this module depends on, push it on the
      // stack (if it hasn't already been marked as visited).
      for (llvm::SetVector<ModuleFile *>::iterator
             M = NextModule->Imports.begin(),
             MEnd = NextModule->Imports.end();
           M != MEnd; ++M) {
        if (State->VisitNumber[(*M)->Index] != VisitNumber) {
          State->Stack.push_back(*M);
          State->VisitNumber[(*M)->Index] = VisitNumber;
        }
      }

      if (State->Stack.empty())
        break;

      // Pop the next module off the stack.
      NextModule = State->Stack.pop_back_val();
    } while (true);
  }

  returnVisitState(std::move(State));
}

#ifndef NDEBUG
namespace llvm {

  template<>
  struct GraphTraits<ModuleManager> {
    using NodeRef = ModuleFile *;
    using ChildIteratorType = llvm::SetVector<ModuleFile *>::const_iterator;
    using nodes_iterator = pointer_iterator<ModuleManager::ModuleConstIterator>;

    static ChildIteratorType child_begin(NodeRef Node) {
      return Node->Imports.begin();
    }

    static ChildIteratorType child_end(NodeRef Node) {
      return Node->Imports.end();
    }

    static nodes_iterator nodes_begin(const ModuleManager &Manager) {
      return nodes_iterator(Manager.begin());
    }

    static nodes_iterator nodes_end(const ModuleManager &Manager) {
      return nodes_iterator(Manager.end());
    }
  };

  template<>
  struct DOTGraphTraits<ModuleManager> : public DefaultDOTGraphTraits {
    explicit DOTGraphTraits(bool IsSimple = false)
        : DefaultDOTGraphTraits(IsSimple) {}

    static bool renderGraphFromBottomUp() { return true; }

    std::string getNodeLabel(ModuleFile *M, const ModuleManager&) {
      return M->ModuleName;
    }
  };

} // namespace llvm

void ModuleManager::viewGraph() {
  llvm::ViewGraph(*this, "Modules");
}
#endif
