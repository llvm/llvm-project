//===--- ASTUnit.cpp - ASTUnit utility ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ASTUnit Implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
using namespace clang;

using llvm::TimeRecord;

namespace {
  class SimpleTimer {
    bool WantTiming;
    TimeRecord Start;
    std::string Output;

  public:
    explicit SimpleTimer(bool WantTiming) : WantTiming(WantTiming) {
      if (WantTiming)
        Start = TimeRecord::getCurrentTime();
    }

    void setOutput(const Twine &Output) {
      if (WantTiming)
        this->Output = Output.str();
    }

    ~SimpleTimer() {
      if (WantTiming) {
        TimeRecord Elapsed = TimeRecord::getCurrentTime();
        Elapsed -= Start;
        llvm::errs() << Output << ':';
        Elapsed.print(Elapsed, llvm::errs());
        llvm::errs() << '\n';
      }
    }
  };
  
  struct OnDiskData {
    /// \brief The file in which the precompiled preamble is stored.
    std::string PreambleFile;

    /// \brief Temporary files that should be removed when the ASTUnit is 
    /// destroyed.
    SmallVector<llvm::sys::Path, 4> TemporaryFiles;
    
    /// \brief Erase temporary files.
    void CleanTemporaryFiles();

    /// \brief Erase the preamble file.
    void CleanPreambleFile();

    /// \brief Erase temporary files and the preamble file.
    void Cleanup();
  };
}

static llvm::sys::SmartMutex<false> &getOnDiskMutex() {
  static llvm::sys::SmartMutex<false> M(/* recursive = */ true);
  return M;
}

static void cleanupOnDiskMapAtExit(void);

typedef llvm::DenseMap<const ASTUnit *, OnDiskData *> OnDiskDataMap;
static OnDiskDataMap &getOnDiskDataMap() {
  static OnDiskDataMap M;
  static bool hasRegisteredAtExit = false;
  if (!hasRegisteredAtExit) {
    hasRegisteredAtExit = true;
    atexit(cleanupOnDiskMapAtExit);
  }
  return M;
}

static void cleanupOnDiskMapAtExit(void) {
  // Use the mutex because there can be an alive thread destroying an ASTUnit.
  llvm::MutexGuard Guard(getOnDiskMutex());
  OnDiskDataMap &M = getOnDiskDataMap();
  for (OnDiskDataMap::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    // We don't worry about freeing the memory associated with OnDiskDataMap.
    // All we care about is erasing stale files.
    I->second->Cleanup();
  }
}

static OnDiskData &getOnDiskData(const ASTUnit *AU) {
  // We require the mutex since we are modifying the structure of the
  // DenseMap.
  llvm::MutexGuard Guard(getOnDiskMutex());
  OnDiskDataMap &M = getOnDiskDataMap();
  OnDiskData *&D = M[AU];
  if (!D)
    D = new OnDiskData();
  return *D;
}

static void erasePreambleFile(const ASTUnit *AU) {
  getOnDiskData(AU).CleanPreambleFile();
}

static void removeOnDiskEntry(const ASTUnit *AU) {
  // We require the mutex since we are modifying the structure of the
  // DenseMap.
  llvm::MutexGuard Guard(getOnDiskMutex());
  OnDiskDataMap &M = getOnDiskDataMap();
  OnDiskDataMap::iterator I = M.find(AU);
  if (I != M.end()) {
    I->second->Cleanup();
    delete I->second;
    M.erase(AU);
  }
}

static void setPreambleFile(const ASTUnit *AU, llvm::StringRef preambleFile) {
  getOnDiskData(AU).PreambleFile = preambleFile;
}

static const std::string &getPreambleFile(const ASTUnit *AU) {
  return getOnDiskData(AU).PreambleFile;  
}

void OnDiskData::CleanTemporaryFiles() {
  for (unsigned I = 0, N = TemporaryFiles.size(); I != N; ++I)
    TemporaryFiles[I].eraseFromDisk();
  TemporaryFiles.clear(); 
}

void OnDiskData::CleanPreambleFile() {
  if (!PreambleFile.empty()) {
    llvm::sys::Path(PreambleFile).eraseFromDisk();
    PreambleFile.clear();
  }
}

void OnDiskData::Cleanup() {
  CleanTemporaryFiles();
  CleanPreambleFile();
}

struct ASTUnit::ASTWriterData {
  SmallString<128> Buffer;
  llvm::BitstreamWriter Stream;
  ASTWriter Writer;

  ASTWriterData() : Stream(Buffer), Writer(Stream) { }
};

void ASTUnit::clearFileLevelDecls() {
  for (FileDeclsTy::iterator
         I = FileDecls.begin(), E = FileDecls.end(); I != E; ++I)
    delete I->second;
  FileDecls.clear();
}

void ASTUnit::CleanTemporaryFiles() {
  getOnDiskData(this).CleanTemporaryFiles();
}

void ASTUnit::addTemporaryFile(const llvm::sys::Path &TempFile) {
  getOnDiskData(this).TemporaryFiles.push_back(TempFile);
}

/// \brief After failing to build a precompiled preamble (due to
/// errors in the source that occurs in the preamble), the number of
/// reparses during which we'll skip even trying to precompile the
/// preamble.
const unsigned DefaultPreambleRebuildInterval = 5;

/// \brief Tracks the number of ASTUnit objects that are currently active.
///
/// Used for debugging purposes only.
static llvm::sys::cas_flag ActiveASTUnitObjects;

ASTUnit::ASTUnit(bool _MainFileIsAST)
  : Reader(0), OnlyLocalDecls(false), CaptureDiagnostics(false),
    MainFileIsAST(_MainFileIsAST), 
    TUKind(TU_Complete), WantTiming(getenv("LIBCLANG_TIMING")),
    OwnsRemappedFileBuffers(true),
    NumStoredDiagnosticsFromDriver(0),
    PreambleRebuildCounter(0), SavedMainFileBuffer(0), PreambleBuffer(0),
    NumWarningsInPreamble(0),
    ShouldCacheCodeCompletionResults(false),
    IncludeBriefCommentsInCodeCompletion(false), UserFilesAreVolatile(false),
    CompletionCacheTopLevelHashValue(0),
    PreambleTopLevelHashValue(0),
    CurrentTopLevelHashValue(0),
    UnsafeToFree(false) { 
  if (getenv("LIBCLANG_OBJTRACKING")) {
    llvm::sys::AtomicIncrement(&ActiveASTUnitObjects);
    fprintf(stderr, "+++ %d translation units\n", ActiveASTUnitObjects);
  }    
}

ASTUnit::~ASTUnit() {
  clearFileLevelDecls();

  // Clean up the temporary files and the preamble file.
  removeOnDiskEntry(this);

  // Free the buffers associated with remapped files. We are required to
  // perform this operation here because we explicitly request that the
  // compiler instance *not* free these buffers for each invocation of the
  // parser.
  if (Invocation.getPtr() && OwnsRemappedFileBuffers) {
    PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
    for (PreprocessorOptions::remapped_file_buffer_iterator
           FB = PPOpts.remapped_file_buffer_begin(),
           FBEnd = PPOpts.remapped_file_buffer_end();
         FB != FBEnd;
         ++FB)
      delete FB->second;
  }
  
  delete SavedMainFileBuffer;
  delete PreambleBuffer;

  ClearCachedCompletionResults();  
  
  if (getenv("LIBCLANG_OBJTRACKING")) {
    llvm::sys::AtomicDecrement(&ActiveASTUnitObjects);
    fprintf(stderr, "--- %d translation units\n", ActiveASTUnitObjects);
  }    
}

void ASTUnit::setPreprocessor(Preprocessor *pp) { PP = pp; }

/// \brief Determine the set of code-completion contexts in which this 
/// declaration should be shown.
static unsigned getDeclShowContexts(NamedDecl *ND,
                                    const LangOptions &LangOpts,
                                    bool &IsNestedNameSpecifier) {
  IsNestedNameSpecifier = false;
  
  if (isa<UsingShadowDecl>(ND))
    ND = dyn_cast<NamedDecl>(ND->getUnderlyingDecl());
  if (!ND)
    return 0;
  
  uint64_t Contexts = 0;
  if (isa<TypeDecl>(ND) || isa<ObjCInterfaceDecl>(ND) || 
      isa<ClassTemplateDecl>(ND) || isa<TemplateTemplateParmDecl>(ND)) {
    // Types can appear in these contexts.
    if (LangOpts.CPlusPlus || !isa<TagDecl>(ND))
      Contexts |= (1LL << CodeCompletionContext::CCC_TopLevel)
               |  (1LL << CodeCompletionContext::CCC_ObjCIvarList)
               |  (1LL << CodeCompletionContext::CCC_ClassStructUnion)
               |  (1LL << CodeCompletionContext::CCC_Statement)
               |  (1LL << CodeCompletionContext::CCC_Type)
               |  (1LL << CodeCompletionContext::CCC_ParenthesizedExpression);

    // In C++, types can appear in expressions contexts (for functional casts).
    if (LangOpts.CPlusPlus)
      Contexts |= (1LL << CodeCompletionContext::CCC_Expression);
    
    // In Objective-C, message sends can send interfaces. In Objective-C++,
    // all types are available due to functional casts.
    if (LangOpts.CPlusPlus || isa<ObjCInterfaceDecl>(ND))
      Contexts |= (1LL << CodeCompletionContext::CCC_ObjCMessageReceiver);
    
    // In Objective-C, you can only be a subclass of another Objective-C class
    if (isa<ObjCInterfaceDecl>(ND))
      Contexts |= (1LL << CodeCompletionContext::CCC_ObjCInterfaceName);

    // Deal with tag names.
    if (isa<EnumDecl>(ND)) {
      Contexts |= (1LL << CodeCompletionContext::CCC_EnumTag);
      
      // Part of the nested-name-specifier in C++0x.
      if (LangOpts.CPlusPlus0x)
        IsNestedNameSpecifier = true;
    } else if (RecordDecl *Record = dyn_cast<RecordDecl>(ND)) {
      if (Record->isUnion())
        Contexts |= (1LL << CodeCompletionContext::CCC_UnionTag);
      else
        Contexts |= (1LL << CodeCompletionContext::CCC_ClassOrStructTag);
      
      if (LangOpts.CPlusPlus)
        IsNestedNameSpecifier = true;
    } else if (isa<ClassTemplateDecl>(ND))
      IsNestedNameSpecifier = true;
  } else if (isa<ValueDecl>(ND) || isa<FunctionTemplateDecl>(ND)) {
    // Values can appear in these contexts.
    Contexts = (1LL << CodeCompletionContext::CCC_Statement)
             | (1LL << CodeCompletionContext::CCC_Expression)
             | (1LL << CodeCompletionContext::CCC_ParenthesizedExpression)
             | (1LL << CodeCompletionContext::CCC_ObjCMessageReceiver);
  } else if (isa<ObjCProtocolDecl>(ND)) {
    Contexts = (1LL << CodeCompletionContext::CCC_ObjCProtocolName);
  } else if (isa<ObjCCategoryDecl>(ND)) {
    Contexts = (1LL << CodeCompletionContext::CCC_ObjCCategoryName);
  } else if (isa<NamespaceDecl>(ND) || isa<NamespaceAliasDecl>(ND)) {
    Contexts = (1LL << CodeCompletionContext::CCC_Namespace);
   
    // Part of the nested-name-specifier.
    IsNestedNameSpecifier = true;
  }
  
  return Contexts;
}

void ASTUnit::CacheCodeCompletionResults() {
  if (!TheSema)
    return;
  
  SimpleTimer Timer(WantTiming);
  Timer.setOutput("Cache global code completions for " + getMainFileName());

  // Clear out the previous results.
  ClearCachedCompletionResults();
  
  // Gather the set of global code completions.
  typedef CodeCompletionResult Result;
  SmallVector<Result, 8> Results;
  CachedCompletionAllocator = new GlobalCodeCompletionAllocator;
  TheSema->GatherGlobalCodeCompletions(*CachedCompletionAllocator,
                                       getCodeCompletionTUInfo(), Results);
  
  // Translate global code completions into cached completions.
  llvm::DenseMap<CanQualType, unsigned> CompletionTypes;
  
  for (unsigned I = 0, N = Results.size(); I != N; ++I) {
    switch (Results[I].Kind) {
    case Result::RK_Declaration: {
      bool IsNestedNameSpecifier = false;
      CachedCodeCompletionResult CachedResult;
      CachedResult.Completion = Results[I].CreateCodeCompletionString(*TheSema,
                                                    *CachedCompletionAllocator,
                                                    getCodeCompletionTUInfo(),
                                          IncludeBriefCommentsInCodeCompletion);
      CachedResult.ShowInContexts = getDeclShowContexts(Results[I].Declaration,
                                                        Ctx->getLangOpts(),
                                                        IsNestedNameSpecifier);
      CachedResult.Priority = Results[I].Priority;
      CachedResult.Kind = Results[I].CursorKind;
      CachedResult.Availability = Results[I].Availability;

      // Keep track of the type of this completion in an ASTContext-agnostic 
      // way.
      QualType UsageType = getDeclUsageType(*Ctx, Results[I].Declaration);
      if (UsageType.isNull()) {
        CachedResult.TypeClass = STC_Void;
        CachedResult.Type = 0;
      } else {
        CanQualType CanUsageType
          = Ctx->getCanonicalType(UsageType.getUnqualifiedType());
        CachedResult.TypeClass = getSimplifiedTypeClass(CanUsageType);

        // Determine whether we have already seen this type. If so, we save
        // ourselves the work of formatting the type string by using the 
        // temporary, CanQualType-based hash table to find the associated value.
        unsigned &TypeValue = CompletionTypes[CanUsageType];
        if (TypeValue == 0) {
          TypeValue = CompletionTypes.size();
          CachedCompletionTypes[QualType(CanUsageType).getAsString()]
            = TypeValue;
        }
        
        CachedResult.Type = TypeValue;
      }
      
      CachedCompletionResults.push_back(CachedResult);
      
      /// Handle nested-name-specifiers in C++.
      if (TheSema->Context.getLangOpts().CPlusPlus && 
          IsNestedNameSpecifier && !Results[I].StartsNestedNameSpecifier) {
        // The contexts in which a nested-name-specifier can appear in C++.
        uint64_t NNSContexts
          = (1LL << CodeCompletionContext::CCC_TopLevel)
          | (1LL << CodeCompletionContext::CCC_ObjCIvarList)
          | (1LL << CodeCompletionContext::CCC_ClassStructUnion)
          | (1LL << CodeCompletionContext::CCC_Statement)
          | (1LL << CodeCompletionContext::CCC_Expression)
          | (1LL << CodeCompletionContext::CCC_ObjCMessageReceiver)
          | (1LL << CodeCompletionContext::CCC_EnumTag)
          | (1LL << CodeCompletionContext::CCC_UnionTag)
          | (1LL << CodeCompletionContext::CCC_ClassOrStructTag)
          | (1LL << CodeCompletionContext::CCC_Type)
          | (1LL << CodeCompletionContext::CCC_PotentiallyQualifiedName)
          | (1LL << CodeCompletionContext::CCC_ParenthesizedExpression);

        if (isa<NamespaceDecl>(Results[I].Declaration) ||
            isa<NamespaceAliasDecl>(Results[I].Declaration))
          NNSContexts |= (1LL << CodeCompletionContext::CCC_Namespace);

        if (unsigned RemainingContexts 
                                = NNSContexts & ~CachedResult.ShowInContexts) {
          // If there any contexts where this completion can be a 
          // nested-name-specifier but isn't already an option, create a 
          // nested-name-specifier completion.
          Results[I].StartsNestedNameSpecifier = true;
          CachedResult.Completion 
            = Results[I].CreateCodeCompletionString(*TheSema,
                                                    *CachedCompletionAllocator,
                                                    getCodeCompletionTUInfo(),
                                        IncludeBriefCommentsInCodeCompletion);
          CachedResult.ShowInContexts = RemainingContexts;
          CachedResult.Priority = CCP_NestedNameSpecifier;
          CachedResult.TypeClass = STC_Void;
          CachedResult.Type = 0;
          CachedCompletionResults.push_back(CachedResult);
        }
      }
      break;
    }
        
    case Result::RK_Keyword:
    case Result::RK_Pattern:
      // Ignore keywords and patterns; we don't care, since they are so
      // easily regenerated.
      break;
      
    case Result::RK_Macro: {
      CachedCodeCompletionResult CachedResult;
      CachedResult.Completion 
        = Results[I].CreateCodeCompletionString(*TheSema,
                                                *CachedCompletionAllocator,
                                                getCodeCompletionTUInfo(),
                                          IncludeBriefCommentsInCodeCompletion);
      CachedResult.ShowInContexts
        = (1LL << CodeCompletionContext::CCC_TopLevel)
        | (1LL << CodeCompletionContext::CCC_ObjCInterface)
        | (1LL << CodeCompletionContext::CCC_ObjCImplementation)
        | (1LL << CodeCompletionContext::CCC_ObjCIvarList)
        | (1LL << CodeCompletionContext::CCC_ClassStructUnion)
        | (1LL << CodeCompletionContext::CCC_Statement)
        | (1LL << CodeCompletionContext::CCC_Expression)
        | (1LL << CodeCompletionContext::CCC_ObjCMessageReceiver)
        | (1LL << CodeCompletionContext::CCC_MacroNameUse)
        | (1LL << CodeCompletionContext::CCC_PreprocessorExpression)
        | (1LL << CodeCompletionContext::CCC_ParenthesizedExpression)
        | (1LL << CodeCompletionContext::CCC_OtherWithMacros);
      
      CachedResult.Priority = Results[I].Priority;
      CachedResult.Kind = Results[I].CursorKind;
      CachedResult.Availability = Results[I].Availability;
      CachedResult.TypeClass = STC_Void;
      CachedResult.Type = 0;
      CachedCompletionResults.push_back(CachedResult);
      break;
    }
    }
  }
  
  // Save the current top-level hash value.
  CompletionCacheTopLevelHashValue = CurrentTopLevelHashValue;
}

void ASTUnit::ClearCachedCompletionResults() {
  CachedCompletionResults.clear();
  CachedCompletionTypes.clear();
  CachedCompletionAllocator = 0;
}

namespace {

/// \brief Gathers information from ASTReader that will be used to initialize
/// a Preprocessor.
class ASTInfoCollector : public ASTReaderListener {
  Preprocessor &PP;
  ASTContext &Context;
  LangOptions &LangOpt;
  HeaderSearch &HSI;
  IntrusiveRefCntPtr<TargetOptions> &TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> &Target;
  unsigned &Counter;

  unsigned NumHeaderInfos;

  bool InitializedLanguage;
public:
  ASTInfoCollector(Preprocessor &PP, ASTContext &Context, LangOptions &LangOpt, 
                   HeaderSearch &HSI, 
                   IntrusiveRefCntPtr<TargetOptions> &TargetOpts,
                   IntrusiveRefCntPtr<TargetInfo> &Target,
                   unsigned &Counter)
    : PP(PP), Context(Context), LangOpt(LangOpt), HSI(HSI), 
      TargetOpts(TargetOpts), Target(Target),
      Counter(Counter), NumHeaderInfos(0),
      InitializedLanguage(false) {}

  virtual bool ReadLanguageOptions(const LangOptions &LangOpts,
                                   bool Complain) {
    if (InitializedLanguage)
      return false;
    
    LangOpt = LangOpts;
    InitializedLanguage = true;
    
    updated();
    return false;
  }

  virtual bool ReadTargetOptions(const TargetOptions &TargetOpts,
                                 bool Complain) {
    // If we've already initialized the target, don't do it again.
    if (Target)
      return false;
    
    this->TargetOpts = new TargetOptions(TargetOpts);
    Target = TargetInfo::CreateTargetInfo(PP.getDiagnostics(), 
                                          *this->TargetOpts);

    updated();
    return false;
  }

  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID) {
    HSI.setHeaderFileInfoForUID(HFI, NumHeaderInfos++);
  }

  virtual void ReadCounter(const serialization::ModuleFile &M, unsigned Value) {
    Counter = Value;
  }

private:
  void updated() {
    if (!Target || !InitializedLanguage)
      return;

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Target->setForcedLangOptions(LangOpt);

    // Initialize the preprocessor.
    PP.Initialize(*Target);

    // Initialize the ASTContext
    Context.InitBuiltinTypes(*Target);
  }
};

class StoredDiagnosticConsumer : public DiagnosticConsumer {
  SmallVectorImpl<StoredDiagnostic> &StoredDiags;
  
public:
  explicit StoredDiagnosticConsumer(
                          SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : StoredDiags(StoredDiags) { }
  
  virtual void HandleDiagnostic(DiagnosticsEngine::Level Level,
                                const Diagnostic &Info);
  
  DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const {
    // Just drop any diagnostics that come from cloned consumers; they'll
    // have different source managers anyway.
    // FIXME: We'd like to be able to capture these somehow, even if it's just
    // file/line/column, because they could occur when parsing module maps or
    // building modules on-demand.
    return new IgnoringDiagConsumer();
  }
};

/// \brief RAII object that optionally captures diagnostics, if
/// there is no diagnostic client to capture them already.
class CaptureDroppedDiagnostics {
  DiagnosticsEngine &Diags;
  StoredDiagnosticConsumer Client;
  DiagnosticConsumer *PreviousClient;

public:
  CaptureDroppedDiagnostics(bool RequestCapture, DiagnosticsEngine &Diags,
                          SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : Diags(Diags), Client(StoredDiags), PreviousClient(0)
  {
    if (RequestCapture || Diags.getClient() == 0) {
      PreviousClient = Diags.takeClient();
      Diags.setClient(&Client);
    }
  }

  ~CaptureDroppedDiagnostics() {
    if (Diags.getClient() == &Client) {
      Diags.takeClient();
      Diags.setClient(PreviousClient);
    }
  }
};

} // anonymous namespace

void StoredDiagnosticConsumer::HandleDiagnostic(DiagnosticsEngine::Level Level,
                                              const Diagnostic &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(Level, Info);

  StoredDiags.push_back(StoredDiagnostic(Level, Info));
}

ASTDeserializationListener *ASTUnit::getDeserializationListener() {
  if (WriterData)
    return &WriterData->Writer;
  return 0;
}

llvm::MemoryBuffer *ASTUnit::getBufferForFile(StringRef Filename,
                                              std::string *ErrorStr) {
  assert(FileMgr);
  return FileMgr->getBufferForFile(Filename, ErrorStr);
}

/// \brief Configure the diagnostics object for use with ASTUnit.
void ASTUnit::ConfigureDiags(IntrusiveRefCntPtr<DiagnosticsEngine> &Diags,
                             const char **ArgBegin, const char **ArgEnd,
                             ASTUnit &AST, bool CaptureDiagnostics) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    DiagnosticConsumer *Client = 0;
    if (CaptureDiagnostics)
      Client = new StoredDiagnosticConsumer(AST.StoredDiagnostics);
    Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions(),
                                                ArgEnd-ArgBegin,
                                                ArgBegin, Client,
                                                /*ShouldOwnClient=*/true,
                                                /*ShouldCloneClient=*/false);
  } else if (CaptureDiagnostics) {
    Diags->setClient(new StoredDiagnosticConsumer(AST.StoredDiagnostics));
  }
}

ASTUnit *ASTUnit::LoadFromASTFile(const std::string &Filename,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                  const FileSystemOptions &FileSystemOpts,
                                  bool OnlyLocalDecls,
                                  RemappedFile *RemappedFiles,
                                  unsigned NumRemappedFiles,
                                  bool CaptureDiagnostics,
                                  bool AllowPCHWithCompilerErrors,
                                  bool UserFilesAreVolatile) {
  OwningPtr<ASTUnit> AST(new ASTUnit(true));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(AST.get());
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.getPtr());

  ConfigureDiags(Diags, 0, 0, *AST, CaptureDiagnostics);

  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->Diagnostics = Diags;
  AST->FileMgr = new FileManager(FileSystemOpts);
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  AST->SourceMgr = new SourceManager(AST->getDiagnostics(),
                                     AST->getFileManager(),
                                     UserFilesAreVolatile);
  AST->HSOpts = new HeaderSearchOptions();
  
  AST->HeaderInfo.reset(new HeaderSearch(AST->HSOpts,
                                         AST->getFileManager(),
                                         AST->getDiagnostics(),
                                         AST->ASTFileLangOpts,
                                         /*Target=*/0));
  
  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    FilenameOrMemBuf fileOrBuf = RemappedFiles[I].second;
    if (const llvm::MemoryBuffer *
          memBuf = fileOrBuf.dyn_cast<const llvm::MemoryBuffer *>()) {
      // Create the file entry for the file that we're mapping from.
      const FileEntry *FromFile
        = AST->getFileManager().getVirtualFile(RemappedFiles[I].first,
                                               memBuf->getBufferSize(),
                                               0);
      if (!FromFile) {
        AST->getDiagnostics().Report(diag::err_fe_remap_missing_from_file)
          << RemappedFiles[I].first;
        delete memBuf;
        continue;
      }
      
      // Override the contents of the "from" file with the contents of
      // the "to" file.
      AST->getSourceManager().overrideFileContents(FromFile, memBuf);

    } else {
      const char *fname = fileOrBuf.get<const char *>();
      const FileEntry *ToFile = AST->FileMgr->getFile(fname);
      if (!ToFile) {
        AST->getDiagnostics().Report(diag::err_fe_remap_missing_to_file)
        << RemappedFiles[I].first << fname;
        continue;
      }

      // Create the file entry for the file that we're mapping from.
      const FileEntry *FromFile
        = AST->getFileManager().getVirtualFile(RemappedFiles[I].first,
                                               ToFile->getSize(),
                                               0);
      if (!FromFile) {
        AST->getDiagnostics().Report(diag::err_fe_remap_missing_from_file)
          << RemappedFiles[I].first;
        delete memBuf;
        continue;
      }
      
      // Override the contents of the "from" file with the contents of
      // the "to" file.
      AST->getSourceManager().overrideFileContents(FromFile, ToFile);
    }
  }
  
  // Gather Info for preprocessor construction later on.

  HeaderSearch &HeaderInfo = *AST->HeaderInfo.get();
  unsigned Counter;

  OwningPtr<ASTReader> Reader;

  AST->PP = new Preprocessor(new PreprocessorOptions(),
                             AST->getDiagnostics(), AST->ASTFileLangOpts,
                             /*Target=*/0, AST->getSourceManager(), HeaderInfo, 
                             *AST, 
                             /*IILookup=*/0,
                             /*OwnsHeaderSearch=*/false,
                             /*DelayInitialization=*/true);
  Preprocessor &PP = *AST->PP;

  AST->Ctx = new ASTContext(AST->ASTFileLangOpts,
                            AST->getSourceManager(),
                            /*Target=*/0,
                            PP.getIdentifierTable(),
                            PP.getSelectorTable(),
                            PP.getBuiltinInfo(),
                            /* size_reserve = */0,
                            /*DelayInitialization=*/true);
  ASTContext &Context = *AST->Ctx;

  bool disableValid = false;
  if (::getenv("LIBCLANG_DISABLE_PCH_VALIDATION"))
    disableValid = true;
  Reader.reset(new ASTReader(PP, Context,
                             /*isysroot=*/"",
                             /*DisableValidation=*/disableValid,
                             AllowPCHWithCompilerErrors));
  
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTReader>
    ReaderCleanup(Reader.get());

  Reader->setListener(new ASTInfoCollector(*AST->PP, Context,
                                           AST->ASTFileLangOpts, HeaderInfo, 
                                           AST->TargetOpts, AST->Target, 
                                           Counter));

  switch (Reader->ReadAST(Filename, serialization::MK_MainFile,
                          ASTReader::ARR_None)) {
  case ASTReader::Success:
    break;

  case ASTReader::Failure:
  case ASTReader::OutOfDate:
  case ASTReader::VersionMismatch:
  case ASTReader::ConfigurationMismatch:
  case ASTReader::HadErrors:
    AST->getDiagnostics().Report(diag::err_fe_unable_to_load_pch);
    return NULL;
  }

  AST->OriginalSourceFile = Reader->getOriginalSourceFile();

  PP.setCounterValue(Counter);

  // Attach the AST reader to the AST context as an external AST
  // source, so that declarations will be deserialized from the
  // AST file as needed.
  ASTReader *ReaderPtr = Reader.get();
  OwningPtr<ExternalASTSource> Source(Reader.take());

  // Unregister the cleanup for ASTReader.  It will get cleaned up
  // by the ASTUnit cleanup.
  ReaderCleanup.unregister();

  Context.setExternalSource(Source);

  // Create an AST consumer, even though it isn't used.
  AST->Consumer.reset(new ASTConsumer);
  
  // Create a semantic analysis object and tell the AST reader about it.
  AST->TheSema.reset(new Sema(PP, Context, *AST->Consumer));
  AST->TheSema->Initialize();
  ReaderPtr->InitializeSema(*AST->TheSema);
  AST->Reader = ReaderPtr;

  return AST.take();
}

namespace {

/// \brief Preprocessor callback class that updates a hash value with the names 
/// of all macros that have been defined by the translation unit.
class MacroDefinitionTrackerPPCallbacks : public PPCallbacks {
  unsigned &Hash;
  
public:
  explicit MacroDefinitionTrackerPPCallbacks(unsigned &Hash) : Hash(Hash) { }
  
  virtual void MacroDefined(const Token &MacroNameTok, const MacroInfo *MI) {
    Hash = llvm::HashString(MacroNameTok.getIdentifierInfo()->getName(), Hash);
  }
};

/// \brief Add the given declaration to the hash of all top-level entities.
void AddTopLevelDeclarationToHash(Decl *D, unsigned &Hash) {
  if (!D)
    return;
  
  DeclContext *DC = D->getDeclContext();
  if (!DC)
    return;
  
  if (!(DC->isTranslationUnit() || DC->getLookupParent()->isTranslationUnit()))
    return;

  if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    if (ND->getIdentifier())
      Hash = llvm::HashString(ND->getIdentifier()->getName(), Hash);
    else if (DeclarationName Name = ND->getDeclName()) {
      std::string NameStr = Name.getAsString();
      Hash = llvm::HashString(NameStr, Hash);
    }
    return;
  }  
}

class TopLevelDeclTrackerConsumer : public ASTConsumer {
  ASTUnit &Unit;
  unsigned &Hash;
  
public:
  TopLevelDeclTrackerConsumer(ASTUnit &_Unit, unsigned &Hash)
    : Unit(_Unit), Hash(Hash) {
    Hash = 0;
  }

  void handleTopLevelDecl(Decl *D) {
    if (!D)
      return;

    // FIXME: Currently ObjC method declarations are incorrectly being
    // reported as top-level declarations, even though their DeclContext
    // is the containing ObjC @interface/@implementation.  This is a
    // fundamental problem in the parser right now.
    if (isa<ObjCMethodDecl>(D))
      return;

    AddTopLevelDeclarationToHash(D, Hash);
    Unit.addTopLevelDecl(D);

    handleFileLevelDecl(D);
  }

  void handleFileLevelDecl(Decl *D) {
    Unit.addFileLevelDecl(D);
    if (NamespaceDecl *NSD = dyn_cast<NamespaceDecl>(D)) {
      for (NamespaceDecl::decl_iterator
             I = NSD->decls_begin(), E = NSD->decls_end(); I != E; ++I)
        handleFileLevelDecl(*I);
    }
  }

  bool HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it)
      handleTopLevelDecl(*it);
    return true;
  }

  // We're not interested in "interesting" decls.
  void HandleInterestingDecl(DeclGroupRef) {}

  void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it)
      handleTopLevelDecl(*it);
  }

  virtual ASTDeserializationListener *GetASTDeserializationListener() {
    return Unit.getDeserializationListener();
  }
};

class TopLevelDeclTrackerAction : public ASTFrontendAction {
public:
  ASTUnit &Unit;

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile) {
    CI.getPreprocessor().addPPCallbacks(
     new MacroDefinitionTrackerPPCallbacks(Unit.getCurrentTopLevelHashValue()));
    return new TopLevelDeclTrackerConsumer(Unit, 
                                           Unit.getCurrentTopLevelHashValue());
  }

public:
  TopLevelDeclTrackerAction(ASTUnit &_Unit) : Unit(_Unit) {}

  virtual bool hasCodeCompletionSupport() const { return false; }
  virtual TranslationUnitKind getTranslationUnitKind()  { 
    return Unit.getTranslationUnitKind(); 
  }
};

class PrecompilePreambleConsumer : public PCHGenerator {
  ASTUnit &Unit;
  unsigned &Hash;                                   
  std::vector<Decl *> TopLevelDecls;
                                     
public:
  PrecompilePreambleConsumer(ASTUnit &Unit, const Preprocessor &PP, 
                             StringRef isysroot, raw_ostream *Out)
    : PCHGenerator(PP, "", 0, isysroot, Out), Unit(Unit),
      Hash(Unit.getCurrentTopLevelHashValue()) {
    Hash = 0;
  }

  virtual bool HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it) {
      Decl *D = *it;
      // FIXME: Currently ObjC method declarations are incorrectly being
      // reported as top-level declarations, even though their DeclContext
      // is the containing ObjC @interface/@implementation.  This is a
      // fundamental problem in the parser right now.
      if (isa<ObjCMethodDecl>(D))
        continue;
      AddTopLevelDeclarationToHash(D, Hash);
      TopLevelDecls.push_back(D);
    }
    return true;
  }

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    PCHGenerator::HandleTranslationUnit(Ctx);
    if (!Unit.getDiagnostics().hasErrorOccurred()) {
      // Translate the top-level declarations we captured during
      // parsing into declaration IDs in the precompiled
      // preamble. This will allow us to deserialize those top-level
      // declarations when requested.
      for (unsigned I = 0, N = TopLevelDecls.size(); I != N; ++I)
        Unit.addTopLevelDeclFromPreamble(
                                      getWriter().getDeclID(TopLevelDecls[I]));
    }
  }
};

class PrecompilePreambleAction : public ASTFrontendAction {
  ASTUnit &Unit;

public:
  explicit PrecompilePreambleAction(ASTUnit &Unit) : Unit(Unit) {}

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile) {
    std::string Sysroot;
    std::string OutputFile;
    raw_ostream *OS = 0;
    if (GeneratePCHAction::ComputeASTConsumerArguments(CI, InFile, Sysroot,
                                                       OutputFile,
                                                       OS))
      return 0;
    
    if (!CI.getFrontendOpts().RelocatablePCH)
      Sysroot.clear();

    CI.getPreprocessor().addPPCallbacks(
     new MacroDefinitionTrackerPPCallbacks(Unit.getCurrentTopLevelHashValue()));
    return new PrecompilePreambleConsumer(Unit, CI.getPreprocessor(), Sysroot, 
                                          OS);
  }

  virtual bool hasCodeCompletionSupport() const { return false; }
  virtual bool hasASTFileSupport() const { return false; }
  virtual TranslationUnitKind getTranslationUnitKind() { return TU_Prefix; }
};

}

static void checkAndRemoveNonDriverDiags(SmallVectorImpl<StoredDiagnostic> &
                                                            StoredDiagnostics) {
  // Get rid of stored diagnostics except the ones from the driver which do not
  // have a source location.
  for (unsigned I = 0; I < StoredDiagnostics.size(); ++I) {
    if (StoredDiagnostics[I].getLocation().isValid()) {
      StoredDiagnostics.erase(StoredDiagnostics.begin()+I);
      --I;
    }
  }
}

static void checkAndSanitizeDiags(SmallVectorImpl<StoredDiagnostic> &
                                                              StoredDiagnostics,
                                  SourceManager &SM) {
  // The stored diagnostic has the old source manager in it; update
  // the locations to refer into the new source manager. Since we've
  // been careful to make sure that the source manager's state
  // before and after are identical, so that we can reuse the source
  // location itself.
  for (unsigned I = 0, N = StoredDiagnostics.size(); I < N; ++I) {
    if (StoredDiagnostics[I].getLocation().isValid()) {
      FullSourceLoc Loc(StoredDiagnostics[I].getLocation(), SM);
      StoredDiagnostics[I].setLocation(Loc);
    }
  }
}

/// Parse the source file into a translation unit using the given compiler
/// invocation, replacing the current translation unit.
///
/// \returns True if a failure occurred that causes the ASTUnit not to
/// contain any translation-unit information, false otherwise.
bool ASTUnit::Parse(llvm::MemoryBuffer *OverrideMainBuffer) {
  delete SavedMainFileBuffer;
  SavedMainFileBuffer = 0;
  
  if (!Invocation) {
    delete OverrideMainBuffer;
    return true;
  }
  
  // Create the compiler instance to use for building the AST.
  OwningPtr<CompilerInstance> Clang(new CompilerInstance());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  IntrusiveRefCntPtr<CompilerInvocation>
    CCInvocation(new CompilerInvocation(*Invocation));

  Clang->setInvocation(CCInvocation.getPtr());
  OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
    
  // Set up diagnostics, capturing any diagnostics that would
  // otherwise be dropped.
  Clang->setDiagnostics(&getDiagnostics());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(Clang->getDiagnostics(),
                   Clang->getTargetOpts()));
  if (!Clang->hasTarget()) {
    delete OverrideMainBuffer;
    return true;
  }

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().setForcedLangOptions(Clang->getLangOpts());
  
  assert(Clang->getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_LLVM_IR &&
         "IR inputs not support here!");

  // Configure the various subsystems.
  // FIXME: Should we retain the previous file manager?
  LangOpts = &Clang->getLangOpts();
  FileSystemOpts = Clang->getFileSystemOpts();
  FileMgr = new FileManager(FileSystemOpts);
  SourceMgr = new SourceManager(getDiagnostics(), *FileMgr,
                                UserFilesAreVolatile);
  TheSema.reset();
  Ctx = 0;
  PP = 0;
  Reader = 0;
  
  // Clear out old caches and data.
  TopLevelDecls.clear();
  clearFileLevelDecls();
  CleanTemporaryFiles();

  if (!OverrideMainBuffer) {
    checkAndRemoveNonDriverDiags(StoredDiagnostics);
    TopLevelDeclsInPreamble.clear();
  }

  // Create a file manager object to provide access to and cache the filesystem.
  Clang->setFileManager(&getFileManager());
  
  // Create the source manager.
  Clang->setSourceManager(&getSourceManager());
  
  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  PreprocessorOptions &PreprocessorOpts = Clang->getPreprocessorOpts();
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile, OverrideMainBuffer);
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PreprocessorOpts.ImplicitPCHInclude = getPreambleFile(this);
    PreprocessorOpts.DisablePCHValidation = true;
    
    // The stored diagnostic has the old source manager in it; update
    // the locations to refer into the new source manager. Since we've
    // been careful to make sure that the source manager's state
    // before and after are identical, so that we can reuse the source
    // location itself.
    checkAndSanitizeDiags(StoredDiagnostics, getSourceManager());

    // Keep track of the override buffer;
    SavedMainFileBuffer = OverrideMainBuffer;
  }
  
  OwningPtr<TopLevelDeclTrackerAction> Act(
    new TopLevelDeclTrackerAction(*this));
    
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<TopLevelDeclTrackerAction>
    ActCleanup(Act.get());

  if (!Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0]))
    goto error;

  if (OverrideMainBuffer) {
    std::string ModName = getPreambleFile(this);
    TranslateStoredDiagnostics(Clang->getModuleManager(), ModName,
                               getSourceManager(), PreambleDiagnostics,
                               StoredDiagnostics);
  }

  if (!Act->Execute())
    goto error;

  transferASTDataFromCompilerInstance(*Clang);
  
  Act->EndSourceFile();

  FailedParseDiagnostics.clear();

  return false;

error:
  // Remove the overridden buffer we used for the preamble.
  if (OverrideMainBuffer) {
    delete OverrideMainBuffer;
    SavedMainFileBuffer = 0;
  }

  // Keep the ownership of the data in the ASTUnit because the client may
  // want to see the diagnostics.
  transferASTDataFromCompilerInstance(*Clang);
  FailedParseDiagnostics.swap(StoredDiagnostics);
  StoredDiagnostics.clear();
  NumStoredDiagnosticsFromDriver = 0;
  return true;
}

/// \brief Simple function to retrieve a path for a preamble precompiled header.
static std::string GetPreamblePCHPath() {
  // FIXME: This is lame; sys::Path should provide this function (in particular,
  // it should know how to find the temporary files dir).
  // FIXME: This is really lame. I copied this code from the Driver!
  // FIXME: This is a hack so that we can override the preamble file during
  // crash-recovery testing, which is the only case where the preamble files
  // are not necessarily cleaned up. 
  const char *TmpFile = ::getenv("CINDEXTEST_PREAMBLE_FILE");
  if (TmpFile)
    return TmpFile;
  
  std::string Error;
  const char *TmpDir = ::getenv("TMPDIR");
  if (!TmpDir)
    TmpDir = ::getenv("TEMP");
  if (!TmpDir)
    TmpDir = ::getenv("TMP");
#ifdef LLVM_ON_WIN32
  if (!TmpDir)
    TmpDir = ::getenv("USERPROFILE");
#endif
  if (!TmpDir)
    TmpDir = "/tmp";
  llvm::sys::Path P(TmpDir);
  P.createDirectoryOnDisk(true);
  P.appendComponent("preamble");
  P.appendSuffix("pch");
  if (P.makeUnique(/*reuse_current=*/false, /*ErrMsg*/0))
    return std::string();
  
  return P.str();
}

/// \brief Compute the preamble for the main file, providing the source buffer
/// that corresponds to the main file along with a pair (bytes, start-of-line)
/// that describes the preamble.
std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> > 
ASTUnit::ComputePreamble(CompilerInvocation &Invocation, 
                         unsigned MaxLines, bool &CreatedBuffer) {
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts = Invocation.getPreprocessorOpts();
  CreatedBuffer = false;
  
  // Try to determine if the main file has been remapped, either from the 
  // command line (to another file) or directly through the compiler invocation
  // (to a memory buffer).
  llvm::MemoryBuffer *Buffer = 0;
  llvm::sys::PathWithStatus MainFilePath(FrontendOpts.Inputs[0].getFile());
  if (const llvm::sys::FileStatus *MainFileStatus = MainFilePath.getFileStatus()) {
    // Check whether there is a file-file remapping of the main file
    for (PreprocessorOptions::remapped_file_iterator
          M = PreprocessorOpts.remapped_file_begin(),
          E = PreprocessorOpts.remapped_file_end();
         M != E;
         ++M) {
      llvm::sys::PathWithStatus MPath(M->first);    
      if (const llvm::sys::FileStatus *MStatus = MPath.getFileStatus()) {
        if (MainFileStatus->uniqueID == MStatus->uniqueID) {
          // We found a remapping. Try to load the resulting, remapped source.
          if (CreatedBuffer) {
            delete Buffer;
            CreatedBuffer = false;
          }
          
          Buffer = getBufferForFile(M->second);
          if (!Buffer)
            return std::make_pair((llvm::MemoryBuffer*)0, 
                                  std::make_pair(0, true));
          CreatedBuffer = true;
        }
      }
    }
    
    // Check whether there is a file-buffer remapping. It supercedes the
    // file-file remapping.
    for (PreprocessorOptions::remapped_file_buffer_iterator
           M = PreprocessorOpts.remapped_file_buffer_begin(),
           E = PreprocessorOpts.remapped_file_buffer_end();
         M != E;
         ++M) {
      llvm::sys::PathWithStatus MPath(M->first);    
      if (const llvm::sys::FileStatus *MStatus = MPath.getFileStatus()) {
        if (MainFileStatus->uniqueID == MStatus->uniqueID) {
          // We found a remapping. 
          if (CreatedBuffer) {
            delete Buffer;
            CreatedBuffer = false;
          }
          
          Buffer = const_cast<llvm::MemoryBuffer *>(M->second);
        }
      }
    }
  }
  
  // If the main source file was not remapped, load it now.
  if (!Buffer) {
    Buffer = getBufferForFile(FrontendOpts.Inputs[0].getFile());
    if (!Buffer)
      return std::make_pair((llvm::MemoryBuffer*)0, std::make_pair(0, true));    
    
    CreatedBuffer = true;
  }
  
  return std::make_pair(Buffer, Lexer::ComputePreamble(Buffer,
                                                       *Invocation.getLangOpts(),
                                                       MaxLines));
}

static llvm::MemoryBuffer *CreatePaddedMainFileBuffer(llvm::MemoryBuffer *Old,
                                                      unsigned NewSize,
                                                      StringRef NewName) {
  llvm::MemoryBuffer *Result
    = llvm::MemoryBuffer::getNewUninitMemBuffer(NewSize, NewName);
  memcpy(const_cast<char*>(Result->getBufferStart()), 
         Old->getBufferStart(), Old->getBufferSize());
  memset(const_cast<char*>(Result->getBufferStart()) + Old->getBufferSize(), 
         ' ', NewSize - Old->getBufferSize() - 1);
  const_cast<char*>(Result->getBufferEnd())[-1] = '\n';  
  
  return Result;
}

/// \brief Attempt to build or re-use a precompiled preamble when (re-)parsing
/// the source file.
///
/// This routine will compute the preamble of the main source file. If a
/// non-trivial preamble is found, it will precompile that preamble into a 
/// precompiled header so that the precompiled preamble can be used to reduce
/// reparsing time. If a precompiled preamble has already been constructed,
/// this routine will determine if it is still valid and, if so, avoid 
/// rebuilding the precompiled preamble.
///
/// \param AllowRebuild When true (the default), this routine is
/// allowed to rebuild the precompiled preamble if it is found to be
/// out-of-date.
///
/// \param MaxLines When non-zero, the maximum number of lines that
/// can occur within the preamble.
///
/// \returns If the precompiled preamble can be used, returns a newly-allocated
/// buffer that should be used in place of the main file when doing so.
/// Otherwise, returns a NULL pointer.
llvm::MemoryBuffer *ASTUnit::getMainBufferWithPrecompiledPreamble(
                              const CompilerInvocation &PreambleInvocationIn,
                                                           bool AllowRebuild,
                                                           unsigned MaxLines) {
  
  IntrusiveRefCntPtr<CompilerInvocation>
    PreambleInvocation(new CompilerInvocation(PreambleInvocationIn));
  FrontendOptions &FrontendOpts = PreambleInvocation->getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts
    = PreambleInvocation->getPreprocessorOpts();

  bool CreatedPreambleBuffer = false;
  std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> > NewPreamble 
    = ComputePreamble(*PreambleInvocation, MaxLines, CreatedPreambleBuffer);

  // If ComputePreamble() Take ownership of the preamble buffer.
  OwningPtr<llvm::MemoryBuffer> OwnedPreambleBuffer;
  if (CreatedPreambleBuffer)
    OwnedPreambleBuffer.reset(NewPreamble.first);

  if (!NewPreamble.second.first) {
    // We couldn't find a preamble in the main source. Clear out the current
    // preamble, if we have one. It's obviously no good any more.
    Preamble.clear();
    erasePreambleFile(this);

    // The next time we actually see a preamble, precompile it.
    PreambleRebuildCounter = 1;
    return 0;
  }
  
  if (!Preamble.empty()) {
    // We've previously computed a preamble. Check whether we have the same
    // preamble now that we did before, and that there's enough space in
    // the main-file buffer within the precompiled preamble to fit the
    // new main file.
    if (Preamble.size() == NewPreamble.second.first &&
        PreambleEndsAtStartOfLine == NewPreamble.second.second &&
        NewPreamble.first->getBufferSize() < PreambleReservedSize-2 &&
        memcmp(Preamble.getBufferStart(), NewPreamble.first->getBufferStart(),
               NewPreamble.second.first) == 0) {
      // The preamble has not changed. We may be able to re-use the precompiled
      // preamble.

      // Check that none of the files used by the preamble have changed.
      bool AnyFileChanged = false;
          
      // First, make a record of those files that have been overridden via
      // remapping or unsaved_files.
      llvm::StringMap<std::pair<off_t, time_t> > OverriddenFiles;
      for (PreprocessorOptions::remapped_file_iterator
                R = PreprocessorOpts.remapped_file_begin(),
             REnd = PreprocessorOpts.remapped_file_end();
           !AnyFileChanged && R != REnd;
           ++R) {
        struct stat StatBuf;
        if (FileMgr->getNoncachedStatValue(R->second, StatBuf)) {
          // If we can't stat the file we're remapping to, assume that something
          // horrible happened.
          AnyFileChanged = true;
          break;
        }
        
        OverriddenFiles[R->first] = std::make_pair(StatBuf.st_size, 
                                                   StatBuf.st_mtime);
      }
      for (PreprocessorOptions::remapped_file_buffer_iterator
                R = PreprocessorOpts.remapped_file_buffer_begin(),
             REnd = PreprocessorOpts.remapped_file_buffer_end();
           !AnyFileChanged && R != REnd;
           ++R) {
        // FIXME: Should we actually compare the contents of file->buffer
        // remappings?
        OverriddenFiles[R->first] = std::make_pair(R->second->getBufferSize(), 
                                                   0);
      }
       
      // Check whether anything has changed.
      for (llvm::StringMap<std::pair<off_t, time_t> >::iterator 
             F = FilesInPreamble.begin(), FEnd = FilesInPreamble.end();
           !AnyFileChanged && F != FEnd; 
           ++F) {
        llvm::StringMap<std::pair<off_t, time_t> >::iterator Overridden
          = OverriddenFiles.find(F->first());
        if (Overridden != OverriddenFiles.end()) {
          // This file was remapped; check whether the newly-mapped file 
          // matches up with the previous mapping.
          if (Overridden->second != F->second)
            AnyFileChanged = true;
          continue;
        }
        
        // The file was not remapped; check whether it has changed on disk.
        struct stat StatBuf;
        if (FileMgr->getNoncachedStatValue(F->first(), StatBuf)) {
          // If we can't stat the file, assume that something horrible happened.
          AnyFileChanged = true;
        } else if (StatBuf.st_size != F->second.first || 
                   StatBuf.st_mtime != F->second.second)
          AnyFileChanged = true;
      }
          
      if (!AnyFileChanged) {
        // Okay! We can re-use the precompiled preamble.

        // Set the state of the diagnostic object to mimic its state
        // after parsing the preamble.
        getDiagnostics().Reset();
        ProcessWarningOptions(getDiagnostics(), 
                              PreambleInvocation->getDiagnosticOpts());
        getDiagnostics().setNumWarnings(NumWarningsInPreamble);

        // Create a version of the main file buffer that is padded to
        // buffer size we reserved when creating the preamble.
        return CreatePaddedMainFileBuffer(NewPreamble.first, 
                                          PreambleReservedSize,
                                          FrontendOpts.Inputs[0].getFile());
      }
    }

    // If we aren't allowed to rebuild the precompiled preamble, just
    // return now.
    if (!AllowRebuild)
      return 0;

    // We can't reuse the previously-computed preamble. Build a new one.
    Preamble.clear();
    PreambleDiagnostics.clear();
    erasePreambleFile(this);
    PreambleRebuildCounter = 1;
  } else if (!AllowRebuild) {
    // We aren't allowed to rebuild the precompiled preamble; just
    // return now.
    return 0;
  }

  // If the preamble rebuild counter > 1, it's because we previously
  // failed to build a preamble and we're not yet ready to try
  // again. Decrement the counter and return a failure.
  if (PreambleRebuildCounter > 1) {
    --PreambleRebuildCounter;
    return 0;
  }

  // Create a temporary file for the precompiled preamble. In rare 
  // circumstances, this can fail.
  std::string PreamblePCHPath = GetPreamblePCHPath();
  if (PreamblePCHPath.empty()) {
    // Try again next time.
    PreambleRebuildCounter = 1;
    return 0;
  }
  
  // We did not previously compute a preamble, or it can't be reused anyway.
  SimpleTimer PreambleTimer(WantTiming);
  PreambleTimer.setOutput("Precompiling preamble");
  
  // Create a new buffer that stores the preamble. The buffer also contains
  // extra space for the original contents of the file (which will be present
  // when we actually parse the file) along with more room in case the file
  // grows.  
  PreambleReservedSize = NewPreamble.first->getBufferSize();
  if (PreambleReservedSize < 4096)
    PreambleReservedSize = 8191;
  else
    PreambleReservedSize *= 2;

  // Save the preamble text for later; we'll need to compare against it for
  // subsequent reparses.
  StringRef MainFilename = PreambleInvocation->getFrontendOpts().Inputs[0].getFile();
  Preamble.assign(FileMgr->getFile(MainFilename),
                  NewPreamble.first->getBufferStart(), 
                  NewPreamble.first->getBufferStart() 
                                                  + NewPreamble.second.first);
  PreambleEndsAtStartOfLine = NewPreamble.second.second;

  delete PreambleBuffer;
  PreambleBuffer
    = llvm::MemoryBuffer::getNewUninitMemBuffer(PreambleReservedSize,
                                                FrontendOpts.Inputs[0].getFile());
  memcpy(const_cast<char*>(PreambleBuffer->getBufferStart()), 
         NewPreamble.first->getBufferStart(), Preamble.size());
  memset(const_cast<char*>(PreambleBuffer->getBufferStart()) + Preamble.size(), 
         ' ', PreambleReservedSize - Preamble.size() - 1);
  const_cast<char*>(PreambleBuffer->getBufferEnd())[-1] = '\n';  
  
  // Remap the main source file to the preamble buffer.
  llvm::sys::PathWithStatus MainFilePath(FrontendOpts.Inputs[0].getFile());
  PreprocessorOpts.addRemappedFile(MainFilePath.str(), PreambleBuffer);
  
  // Tell the compiler invocation to generate a temporary precompiled header.
  FrontendOpts.ProgramAction = frontend::GeneratePCH;
  // FIXME: Generate the precompiled header into memory?
  FrontendOpts.OutputFile = PreamblePCHPath;
  PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
  PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  
  // Create the compiler instance to use for building the precompiled preamble.
  OwningPtr<CompilerInstance> Clang(new CompilerInstance());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  Clang->setInvocation(&*PreambleInvocation);
  OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
  
  // Set up diagnostics, capturing all of the diagnostics produced.
  Clang->setDiagnostics(&getDiagnostics());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(Clang->getDiagnostics(),
                                                Clang->getTargetOpts()));
  if (!Clang->hasTarget()) {
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    return 0;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().setForcedLangOptions(Clang->getLangOpts());
  
  assert(Clang->getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_LLVM_IR &&
         "IR inputs not support here!");
  
  // Clear out old caches and data.
  getDiagnostics().Reset();
  ProcessWarningOptions(getDiagnostics(), Clang->getDiagnosticOpts());
  checkAndRemoveNonDriverDiags(StoredDiagnostics);
  TopLevelDecls.clear();
  TopLevelDeclsInPreamble.clear();
  
  // Create a file manager object to provide access to and cache the filesystem.
  Clang->setFileManager(new FileManager(Clang->getFileSystemOpts()));
  
  // Create the source manager.
  Clang->setSourceManager(new SourceManager(getDiagnostics(),
                                            Clang->getFileManager()));
  
  OwningPtr<PrecompilePreambleAction> Act;
  Act.reset(new PrecompilePreambleAction(*this));
  if (!Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0])) {
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    return 0;
  }
  
  Act->Execute();
  Act->EndSourceFile();

  if (Diagnostics->hasErrorOccurred()) {
    // There were errors parsing the preamble, so no precompiled header was
    // generated. Forget that we even tried.
    // FIXME: Should we leave a note for ourselves to try again?
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    TopLevelDeclsInPreamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    return 0;
  }
  
  // Transfer any diagnostics generated when parsing the preamble into the set
  // of preamble diagnostics.
  PreambleDiagnostics.clear();
  PreambleDiagnostics.insert(PreambleDiagnostics.end(), 
                            stored_diag_afterDriver_begin(), stored_diag_end());
  checkAndRemoveNonDriverDiags(StoredDiagnostics);
  
  // Keep track of the preamble we precompiled.
  setPreambleFile(this, FrontendOpts.OutputFile);
  NumWarningsInPreamble = getDiagnostics().getNumWarnings();
  
  // Keep track of all of the files that the source manager knows about,
  // so we can verify whether they have changed or not.
  FilesInPreamble.clear();
  SourceManager &SourceMgr = Clang->getSourceManager();
  const llvm::MemoryBuffer *MainFileBuffer
    = SourceMgr.getBuffer(SourceMgr.getMainFileID());
  for (SourceManager::fileinfo_iterator F = SourceMgr.fileinfo_begin(),
                                     FEnd = SourceMgr.fileinfo_end();
       F != FEnd;
       ++F) {
    const FileEntry *File = F->second->OrigEntry;
    if (!File || F->second->getRawBuffer() == MainFileBuffer)
      continue;
    
    FilesInPreamble[File->getName()]
      = std::make_pair(F->second->getSize(), File->getModificationTime());
  }
  
  PreambleRebuildCounter = 1;
  PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
  
  // If the hash of top-level entities differs from the hash of the top-level
  // entities the last time we rebuilt the preamble, clear out the completion
  // cache.
  if (CurrentTopLevelHashValue != PreambleTopLevelHashValue) {
    CompletionCacheTopLevelHashValue = 0;
    PreambleTopLevelHashValue = CurrentTopLevelHashValue;
  }
  
  return CreatePaddedMainFileBuffer(NewPreamble.first, 
                                    PreambleReservedSize,
                                    FrontendOpts.Inputs[0].getFile());
}

void ASTUnit::RealizeTopLevelDeclsFromPreamble() {
  std::vector<Decl *> Resolved;
  Resolved.reserve(TopLevelDeclsInPreamble.size());
  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (unsigned I = 0, N = TopLevelDeclsInPreamble.size(); I != N; ++I) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    Decl *D = Source.GetExternalDecl(TopLevelDeclsInPreamble[I]);
    if (D)
      Resolved.push_back(D);
  }
  TopLevelDeclsInPreamble.clear();
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());
}

void ASTUnit::transferASTDataFromCompilerInstance(CompilerInstance &CI) {
  // Steal the created target, context, and preprocessor.
  TheSema.reset(CI.takeSema());
  Consumer.reset(CI.takeASTConsumer());
  Ctx = &CI.getASTContext();
  PP = &CI.getPreprocessor();
  CI.setSourceManager(0);
  CI.setFileManager(0);
  Target = &CI.getTarget();
  Reader = CI.getModuleManager();
}

StringRef ASTUnit::getMainFileName() const {
  return Invocation->getFrontendOpts().Inputs[0].getFile();
}

ASTUnit *ASTUnit::create(CompilerInvocation *CI,
                         IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                         bool CaptureDiagnostics,
                         bool UserFilesAreVolatile) {
  OwningPtr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, 0, 0, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  AST->Invocation = CI;
  AST->FileSystemOpts = CI->getFileSystemOpts();
  AST->FileMgr = new FileManager(AST->FileSystemOpts);
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  AST->SourceMgr = new SourceManager(AST->getDiagnostics(), *AST->FileMgr,
                                     UserFilesAreVolatile);

  return AST.take();
}

ASTUnit *ASTUnit::LoadFromCompilerInvocationAction(CompilerInvocation *CI,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                             ASTFrontendAction *Action,
                                             ASTUnit *Unit,
                                             bool Persistent,
                                             StringRef ResourceFilesPath,
                                             bool OnlyLocalDecls,
                                             bool CaptureDiagnostics,
                                             bool PrecompilePreamble,
                                             bool CacheCodeCompletionResults,
                                    bool IncludeBriefCommentsInCodeCompletion,
                                             bool UserFilesAreVolatile,
                                             OwningPtr<ASTUnit> *ErrAST) {
  assert(CI && "A CompilerInvocation is required");

  OwningPtr<ASTUnit> OwnAST;
  ASTUnit *AST = Unit;
  if (!AST) {
    // Create the AST unit.
    OwnAST.reset(create(CI, Diags, CaptureDiagnostics, UserFilesAreVolatile));
    AST = OwnAST.get();
  }
  
  if (!ResourceFilesPath.empty()) {
    // Override the resources path.
    CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;
  }
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  if (PrecompilePreamble)
    AST->PreambleRebuildCounter = 2;
  AST->TUKind = Action ? Action->getTranslationUnitKind() : TU_Complete;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->IncludeBriefCommentsInCodeCompletion
    = IncludeBriefCommentsInCodeCompletion;

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(OwnAST.get());
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.getPtr());

  // We'll manage file buffers ourselves.
  CI->getPreprocessorOpts().RetainRemappedFileBuffers = true;
  CI->getFrontendOpts().DisableFree = false;
  ProcessWarningOptions(AST->getDiagnostics(), CI->getDiagnosticOpts());

  // Create the compiler instance to use for building the AST.
  OwningPtr<CompilerInstance> Clang(new CompilerInstance());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  Clang->setInvocation(CI);
  AST->OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
    
  // Set up diagnostics, capturing any diagnostics that would
  // otherwise be dropped.
  Clang->setDiagnostics(&AST->getDiagnostics());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(Clang->getDiagnostics(),
                   Clang->getTargetOpts()));
  if (!Clang->hasTarget())
    return 0;

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().setForcedLangOptions(Clang->getLangOpts());
  
  assert(Clang->getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_LLVM_IR &&
         "IR inputs not supported here!");

  // Configure the various subsystems.
  AST->TheSema.reset();
  AST->Ctx = 0;
  AST->PP = 0;
  AST->Reader = 0;
  
  // Create a file manager object to provide access to and cache the filesystem.
  Clang->setFileManager(&AST->getFileManager());
  
  // Create the source manager.
  Clang->setSourceManager(&AST->getSourceManager());

  ASTFrontendAction *Act = Action;

  OwningPtr<TopLevelDeclTrackerAction> TrackerAct;
  if (!Act) {
    TrackerAct.reset(new TopLevelDeclTrackerAction(*AST));
    Act = TrackerAct.get();
  }

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<TopLevelDeclTrackerAction>
    ActCleanup(TrackerAct.get());

  if (!Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0])) {
    AST->transferASTDataFromCompilerInstance(*Clang);
    if (OwnAST && ErrAST)
      ErrAST->swap(OwnAST);

    return 0;
  }

  if (Persistent && !TrackerAct) {
    Clang->getPreprocessor().addPPCallbacks(
     new MacroDefinitionTrackerPPCallbacks(AST->getCurrentTopLevelHashValue()));
    std::vector<ASTConsumer*> Consumers;
    if (Clang->hasASTConsumer())
      Consumers.push_back(Clang->takeASTConsumer());
    Consumers.push_back(new TopLevelDeclTrackerConsumer(*AST,
                                           AST->getCurrentTopLevelHashValue()));
    Clang->setASTConsumer(new MultiplexConsumer(Consumers));
  }
  if (!Act->Execute()) {
    AST->transferASTDataFromCompilerInstance(*Clang);
    if (OwnAST && ErrAST)
      ErrAST->swap(OwnAST);

    return 0;
  }

  // Steal the created target, context, and preprocessor.
  AST->transferASTDataFromCompilerInstance(*Clang);
  
  Act->EndSourceFile();

  if (OwnAST)
    return OwnAST.take();
  else
    return AST;
}

bool ASTUnit::LoadFromCompilerInvocation(bool PrecompilePreamble) {
  if (!Invocation)
    return true;
  
  // We'll manage file buffers ourselves.
  Invocation->getPreprocessorOpts().RetainRemappedFileBuffers = true;
  Invocation->getFrontendOpts().DisableFree = false;
  ProcessWarningOptions(getDiagnostics(), Invocation->getDiagnosticOpts());

  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (PrecompilePreamble) {
    PreambleRebuildCounter = 2;
    OverrideMainBuffer
      = getMainBufferWithPrecompiledPreamble(*Invocation);
  }
  
  SimpleTimer ParsingTimer(WantTiming);
  ParsingTimer.setOutput("Parsing " + getMainFileName());
  
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<llvm::MemoryBuffer>
    MemBufferCleanup(OverrideMainBuffer);
  
  return Parse(OverrideMainBuffer);
}

ASTUnit *ASTUnit::LoadFromCompilerInvocation(CompilerInvocation *CI,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                             bool OnlyLocalDecls,
                                             bool CaptureDiagnostics,
                                             bool PrecompilePreamble,
                                             TranslationUnitKind TUKind,
                                             bool CacheCodeCompletionResults,
                                    bool IncludeBriefCommentsInCodeCompletion,
                                             bool UserFilesAreVolatile) {
  // Create the AST unit.
  OwningPtr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, 0, 0, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->TUKind = TUKind;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->IncludeBriefCommentsInCodeCompletion
    = IncludeBriefCommentsInCodeCompletion;
  AST->Invocation = CI;
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(AST.get());
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.getPtr());

  return AST->LoadFromCompilerInvocation(PrecompilePreamble)? 0 : AST.take();
}

ASTUnit *ASTUnit::LoadFromCommandLine(const char **ArgBegin,
                                      const char **ArgEnd,
                                    IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                      StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls,
                                      bool CaptureDiagnostics,
                                      RemappedFile *RemappedFiles,
                                      unsigned NumRemappedFiles,
                                      bool RemappedFilesKeepOriginalName,
                                      bool PrecompilePreamble,
                                      TranslationUnitKind TUKind,
                                      bool CacheCodeCompletionResults,
                                      bool IncludeBriefCommentsInCodeCompletion,
                                      bool AllowPCHWithCompilerErrors,
                                      bool SkipFunctionBodies,
                                      bool UserFilesAreVolatile,
                                      bool ForSerialization,
                                      OwningPtr<ASTUnit> *ErrAST) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions(),
                                                ArgEnd - ArgBegin,
                                                ArgBegin);
  }

  SmallVector<StoredDiagnostic, 4> StoredDiagnostics;
  
  IntrusiveRefCntPtr<CompilerInvocation> CI;

  {

    CaptureDroppedDiagnostics Capture(CaptureDiagnostics, *Diags, 
                                      StoredDiagnostics);

    CI = clang::createInvocationFromCommandLine(
                                           llvm::makeArrayRef(ArgBegin, ArgEnd),
                                           Diags);
    if (!CI)
      return 0;
  }

  // Override any files that need remapping
  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    FilenameOrMemBuf fileOrBuf = RemappedFiles[I].second;
    if (const llvm::MemoryBuffer *
            memBuf = fileOrBuf.dyn_cast<const llvm::MemoryBuffer *>()) {
      CI->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first, memBuf);
    } else {
      const char *fname = fileOrBuf.get<const char *>();
      CI->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first, fname);
    }
  }
  PreprocessorOptions &PPOpts = CI->getPreprocessorOpts();
  PPOpts.RemappedFilesKeepOriginalName = RemappedFilesKeepOriginalName;
  PPOpts.AllowPCHWithCompilerErrors = AllowPCHWithCompilerErrors;
  
  // Override the resources path.
  CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;

  CI->getFrontendOpts().SkipFunctionBodies = SkipFunctionBodies;

  // Create the AST unit.
  OwningPtr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, ArgBegin, ArgEnd, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  Diags = 0; // Zero out now to ease cleanup during crash recovery.
  AST->FileSystemOpts = CI->getFileSystemOpts();
  AST->FileMgr = new FileManager(AST->FileSystemOpts);
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->TUKind = TUKind;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->IncludeBriefCommentsInCodeCompletion
    = IncludeBriefCommentsInCodeCompletion;
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  AST->NumStoredDiagnosticsFromDriver = StoredDiagnostics.size();
  AST->StoredDiagnostics.swap(StoredDiagnostics);
  AST->Invocation = CI;
  if (ForSerialization)
    AST->WriterData.reset(new ASTWriterData());
  CI = 0; // Zero out now to ease cleanup during crash recovery.
  
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(AST.get());

  if (AST->LoadFromCompilerInvocation(PrecompilePreamble)) {
    // Some error occurred, if caller wants to examine diagnostics, pass it the
    // ASTUnit.
    if (ErrAST) {
      AST->StoredDiagnostics.swap(AST->FailedParseDiagnostics);
      ErrAST->swap(AST);
    }
    return 0;
  }

  return AST.take();
}

bool ASTUnit::Reparse(RemappedFile *RemappedFiles, unsigned NumRemappedFiles) {
  if (!Invocation)
    return true;

  clearFileLevelDecls();
  
  SimpleTimer ParsingTimer(WantTiming);
  ParsingTimer.setOutput("Reparsing " + getMainFileName());

  // Remap files.
  PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
  for (PreprocessorOptions::remapped_file_buffer_iterator 
         R = PPOpts.remapped_file_buffer_begin(),
         REnd = PPOpts.remapped_file_buffer_end();
       R != REnd; 
       ++R) {
    delete R->second;
  }
  Invocation->getPreprocessorOpts().clearRemappedFiles();
  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    FilenameOrMemBuf fileOrBuf = RemappedFiles[I].second;
    if (const llvm::MemoryBuffer *
            memBuf = fileOrBuf.dyn_cast<const llvm::MemoryBuffer *>()) {
      Invocation->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                                        memBuf);
    } else {
      const char *fname = fileOrBuf.get<const char *>();
      Invocation->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                                        fname);
    }
  }
  
  // If we have a preamble file lying around, or if we might try to
  // build a precompiled preamble, do so now.
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (!getPreambleFile(this).empty() || PreambleRebuildCounter > 0)
    OverrideMainBuffer = getMainBufferWithPrecompiledPreamble(*Invocation);
    
  // Clear out the diagnostics state.
  getDiagnostics().Reset();
  ProcessWarningOptions(getDiagnostics(), Invocation->getDiagnosticOpts());
  if (OverrideMainBuffer)
    getDiagnostics().setNumWarnings(NumWarningsInPreamble);

  // Parse the sources
  bool Result = Parse(OverrideMainBuffer);
  
  // If we're caching global code-completion results, and the top-level 
  // declarations have changed, clear out the code-completion cache.
  if (!Result && ShouldCacheCodeCompletionResults &&
      CurrentTopLevelHashValue != CompletionCacheTopLevelHashValue)
    CacheCodeCompletionResults();

  // We now need to clear out the completion info related to this translation
  // unit; it'll be recreated if necessary.
  CCTUInfo.reset();
  
  return Result;
}

//----------------------------------------------------------------------------//
// Code completion
//----------------------------------------------------------------------------//

namespace {
  /// \brief Code completion consumer that combines the cached code-completion
  /// results from an ASTUnit with the code-completion results provided to it,
  /// then passes the result on to 
  class AugmentedCodeCompleteConsumer : public CodeCompleteConsumer {
    uint64_t NormalContexts;
    ASTUnit &AST;
    CodeCompleteConsumer &Next;
    
  public:
    AugmentedCodeCompleteConsumer(ASTUnit &AST, CodeCompleteConsumer &Next,
                                  const CodeCompleteOptions &CodeCompleteOpts)
      : CodeCompleteConsumer(CodeCompleteOpts, Next.isOutputBinary()),
        AST(AST), Next(Next)
    { 
      // Compute the set of contexts in which we will look when we don't have
      // any information about the specific context.
      NormalContexts 
        = (1LL << CodeCompletionContext::CCC_TopLevel)
        | (1LL << CodeCompletionContext::CCC_ObjCInterface)
        | (1LL << CodeCompletionContext::CCC_ObjCImplementation)
        | (1LL << CodeCompletionContext::CCC_ObjCIvarList)
        | (1LL << CodeCompletionContext::CCC_Statement)
        | (1LL << CodeCompletionContext::CCC_Expression)
        | (1LL << CodeCompletionContext::CCC_ObjCMessageReceiver)
        | (1LL << CodeCompletionContext::CCC_DotMemberAccess)
        | (1LL << CodeCompletionContext::CCC_ArrowMemberAccess)
        | (1LL << CodeCompletionContext::CCC_ObjCPropertyAccess)
        | (1LL << CodeCompletionContext::CCC_ObjCProtocolName)
        | (1LL << CodeCompletionContext::CCC_ParenthesizedExpression)
        | (1LL << CodeCompletionContext::CCC_Recovery);

      if (AST.getASTContext().getLangOpts().CPlusPlus)
        NormalContexts |= (1LL << CodeCompletionContext::CCC_EnumTag)
                       |  (1LL << CodeCompletionContext::CCC_UnionTag)
                       |  (1LL << CodeCompletionContext::CCC_ClassOrStructTag);
    }
    
    virtual void ProcessCodeCompleteResults(Sema &S, 
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults);
    
    virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                           OverloadCandidate *Candidates,
                                           unsigned NumCandidates) { 
      Next.ProcessOverloadCandidates(S, CurrentArg, Candidates, NumCandidates);
    }
    
    virtual CodeCompletionAllocator &getAllocator() {
      return Next.getAllocator();
    }

    virtual CodeCompletionTUInfo &getCodeCompletionTUInfo() {
      return Next.getCodeCompletionTUInfo();
    }
  };
}

/// \brief Helper function that computes which global names are hidden by the
/// local code-completion results.
static void CalculateHiddenNames(const CodeCompletionContext &Context,
                                 CodeCompletionResult *Results,
                                 unsigned NumResults,
                                 ASTContext &Ctx,
                          llvm::StringSet<llvm::BumpPtrAllocator> &HiddenNames){
  bool OnlyTagNames = false;
  switch (Context.getKind()) {
  case CodeCompletionContext::CCC_Recovery:
  case CodeCompletionContext::CCC_TopLevel:
  case CodeCompletionContext::CCC_ObjCInterface:
  case CodeCompletionContext::CCC_ObjCImplementation:
  case CodeCompletionContext::CCC_ObjCIvarList:
  case CodeCompletionContext::CCC_ClassStructUnion:
  case CodeCompletionContext::CCC_Statement:
  case CodeCompletionContext::CCC_Expression:
  case CodeCompletionContext::CCC_ObjCMessageReceiver:
  case CodeCompletionContext::CCC_DotMemberAccess:
  case CodeCompletionContext::CCC_ArrowMemberAccess:
  case CodeCompletionContext::CCC_ObjCPropertyAccess:
  case CodeCompletionContext::CCC_Namespace:
  case CodeCompletionContext::CCC_Type:
  case CodeCompletionContext::CCC_Name:
  case CodeCompletionContext::CCC_PotentiallyQualifiedName:
  case CodeCompletionContext::CCC_ParenthesizedExpression:
  case CodeCompletionContext::CCC_ObjCInterfaceName:
    break;
    
  case CodeCompletionContext::CCC_EnumTag:
  case CodeCompletionContext::CCC_UnionTag:
  case CodeCompletionContext::CCC_ClassOrStructTag:
    OnlyTagNames = true;
    break;
    
  case CodeCompletionContext::CCC_ObjCProtocolName:
  case CodeCompletionContext::CCC_MacroName:
  case CodeCompletionContext::CCC_MacroNameUse:
  case CodeCompletionContext::CCC_PreprocessorExpression:
  case CodeCompletionContext::CCC_PreprocessorDirective:
  case CodeCompletionContext::CCC_NaturalLanguage:
  case CodeCompletionContext::CCC_SelectorName:
  case CodeCompletionContext::CCC_TypeQualifiers:
  case CodeCompletionContext::CCC_Other:
  case CodeCompletionContext::CCC_OtherWithMacros:
  case CodeCompletionContext::CCC_ObjCInstanceMessage:
  case CodeCompletionContext::CCC_ObjCClassMessage:
  case CodeCompletionContext::CCC_ObjCCategoryName:
    // We're looking for nothing, or we're looking for names that cannot
    // be hidden.
    return;
  }
  
  typedef CodeCompletionResult Result;
  for (unsigned I = 0; I != NumResults; ++I) {
    if (Results[I].Kind != Result::RK_Declaration)
      continue;
    
    unsigned IDNS
      = Results[I].Declaration->getUnderlyingDecl()->getIdentifierNamespace();

    bool Hiding = false;
    if (OnlyTagNames)
      Hiding = (IDNS & Decl::IDNS_Tag);
    else {
      unsigned HiddenIDNS = (Decl::IDNS_Type | Decl::IDNS_Member | 
                             Decl::IDNS_Namespace | Decl::IDNS_Ordinary |
                             Decl::IDNS_NonMemberOperator);
      if (Ctx.getLangOpts().CPlusPlus)
        HiddenIDNS |= Decl::IDNS_Tag;
      Hiding = (IDNS & HiddenIDNS);
    }
  
    if (!Hiding)
      continue;
    
    DeclarationName Name = Results[I].Declaration->getDeclName();
    if (IdentifierInfo *Identifier = Name.getAsIdentifierInfo())
      HiddenNames.insert(Identifier->getName());
    else
      HiddenNames.insert(Name.getAsString());
  }
}


void AugmentedCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &S,
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults) { 
  // Merge the results we were given with the results we cached.
  bool AddedResult = false;
  uint64_t InContexts =
      Context.getKind() == CodeCompletionContext::CCC_Recovery
        ? NormalContexts : (1LL << Context.getKind());
  // Contains the set of names that are hidden by "local" completion results.
  llvm::StringSet<llvm::BumpPtrAllocator> HiddenNames;
  typedef CodeCompletionResult Result;
  SmallVector<Result, 8> AllResults;
  for (ASTUnit::cached_completion_iterator 
            C = AST.cached_completion_begin(),
         CEnd = AST.cached_completion_end();
       C != CEnd; ++C) {
    // If the context we are in matches any of the contexts we are 
    // interested in, we'll add this result.
    if ((C->ShowInContexts & InContexts) == 0)
      continue;
    
    // If we haven't added any results previously, do so now.
    if (!AddedResult) {
      CalculateHiddenNames(Context, Results, NumResults, S.Context, 
                           HiddenNames);
      AllResults.insert(AllResults.end(), Results, Results + NumResults);
      AddedResult = true;
    }
    
    // Determine whether this global completion result is hidden by a local
    // completion result. If so, skip it.
    if (C->Kind != CXCursor_MacroDefinition &&
        HiddenNames.count(C->Completion->getTypedText()))
      continue;
    
    // Adjust priority based on similar type classes.
    unsigned Priority = C->Priority;
    CodeCompletionString *Completion = C->Completion;
    if (!Context.getPreferredType().isNull()) {
      if (C->Kind == CXCursor_MacroDefinition) {
        Priority = getMacroUsagePriority(C->Completion->getTypedText(),
                                         S.getLangOpts(),
                               Context.getPreferredType()->isAnyPointerType());        
      } else if (C->Type) {
        CanQualType Expected
          = S.Context.getCanonicalType(
                               Context.getPreferredType().getUnqualifiedType());
        SimplifiedTypeClass ExpectedSTC = getSimplifiedTypeClass(Expected);
        if (ExpectedSTC == C->TypeClass) {
          // We know this type is similar; check for an exact match.
          llvm::StringMap<unsigned> &CachedCompletionTypes
            = AST.getCachedCompletionTypes();
          llvm::StringMap<unsigned>::iterator Pos
            = CachedCompletionTypes.find(QualType(Expected).getAsString());
          if (Pos != CachedCompletionTypes.end() && Pos->second == C->Type)
            Priority /= CCF_ExactTypeMatch;
          else
            Priority /= CCF_SimilarTypeMatch;
        }
      }
    }
    
    // Adjust the completion string, if required.
    if (C->Kind == CXCursor_MacroDefinition &&
        Context.getKind() == CodeCompletionContext::CCC_MacroNameUse) {
      // Create a new code-completion string that just contains the
      // macro name, without its arguments.
      CodeCompletionBuilder Builder(getAllocator(), getCodeCompletionTUInfo(),
                                    CCP_CodePattern, C->Availability);
      Builder.AddTypedTextChunk(C->Completion->getTypedText());
      Priority = CCP_CodePattern;
      Completion = Builder.TakeString();
    }
    
    AllResults.push_back(Result(Completion, Priority, C->Kind,
                                C->Availability));
  }
  
  // If we did not add any cached completion results, just forward the
  // results we were given to the next consumer.
  if (!AddedResult) {
    Next.ProcessCodeCompleteResults(S, Context, Results, NumResults);
    return;
  }
  
  Next.ProcessCodeCompleteResults(S, Context, AllResults.data(),
                                  AllResults.size());
}



void ASTUnit::CodeComplete(StringRef File, unsigned Line, unsigned Column,
                           RemappedFile *RemappedFiles, 
                           unsigned NumRemappedFiles,
                           bool IncludeMacros, 
                           bool IncludeCodePatterns,
                           bool IncludeBriefComments,
                           CodeCompleteConsumer &Consumer,
                           DiagnosticsEngine &Diag, LangOptions &LangOpts,
                           SourceManager &SourceMgr, FileManager &FileMgr,
                   SmallVectorImpl<StoredDiagnostic> &StoredDiagnostics,
             SmallVectorImpl<const llvm::MemoryBuffer *> &OwnedBuffers) {
  if (!Invocation)
    return;

  SimpleTimer CompletionTimer(WantTiming);
  CompletionTimer.setOutput("Code completion @ " + File + ":" +
                            Twine(Line) + ":" + Twine(Column));

  IntrusiveRefCntPtr<CompilerInvocation>
    CCInvocation(new CompilerInvocation(*Invocation));

  FrontendOptions &FrontendOpts = CCInvocation->getFrontendOpts();
  CodeCompleteOptions &CodeCompleteOpts = FrontendOpts.CodeCompleteOpts;
  PreprocessorOptions &PreprocessorOpts = CCInvocation->getPreprocessorOpts();

  CodeCompleteOpts.IncludeMacros = IncludeMacros &&
                                   CachedCompletionResults.empty();
  CodeCompleteOpts.IncludeCodePatterns = IncludeCodePatterns;
  CodeCompleteOpts.IncludeGlobals = CachedCompletionResults.empty();
  CodeCompleteOpts.IncludeBriefComments = IncludeBriefComments;

  assert(IncludeBriefComments == this->IncludeBriefCommentsInCodeCompletion);

  FrontendOpts.CodeCompletionAt.FileName = File;
  FrontendOpts.CodeCompletionAt.Line = Line;
  FrontendOpts.CodeCompletionAt.Column = Column;

  // Set the language options appropriately.
  LangOpts = *CCInvocation->getLangOpts();

  OwningPtr<CompilerInstance> Clang(new CompilerInstance());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  Clang->setInvocation(&*CCInvocation);
  OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
    
  // Set up diagnostics, capturing any diagnostics produced.
  Clang->setDiagnostics(&Diag);
  ProcessWarningOptions(Diag, CCInvocation->getDiagnosticOpts());
  CaptureDroppedDiagnostics Capture(true, 
                                    Clang->getDiagnostics(), 
                                    StoredDiagnostics);
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(Clang->getDiagnostics(),
                                               Clang->getTargetOpts()));
  if (!Clang->hasTarget()) {
    Clang->setInvocation(0);
    return;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().setForcedLangOptions(Clang->getLangOpts());
  
  assert(Clang->getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_LLVM_IR &&
         "IR inputs not support here!");

  
  // Use the source and file managers that we were given.
  Clang->setFileManager(&FileMgr);
  Clang->setSourceManager(&SourceMgr);

  // Remap files.
  PreprocessorOpts.clearRemappedFiles();
  PreprocessorOpts.RetainRemappedFileBuffers = true;
  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    FilenameOrMemBuf fileOrBuf = RemappedFiles[I].second;
    if (const llvm::MemoryBuffer *
            memBuf = fileOrBuf.dyn_cast<const llvm::MemoryBuffer *>()) {
      PreprocessorOpts.addRemappedFile(RemappedFiles[I].first, memBuf);
      OwnedBuffers.push_back(memBuf);
    } else {
      const char *fname = fileOrBuf.get<const char *>();
      PreprocessorOpts.addRemappedFile(RemappedFiles[I].first, fname);
    }
  }
  
  // Use the code completion consumer we were given, but adding any cached
  // code-completion results.
  AugmentedCodeCompleteConsumer *AugmentedConsumer
    = new AugmentedCodeCompleteConsumer(*this, Consumer, CodeCompleteOpts);
  Clang->setCodeCompletionConsumer(AugmentedConsumer);

  // If we have a precompiled preamble, try to use it. We only allow
  // the use of the precompiled preamble if we're if the completion
  // point is within the main file, after the end of the precompiled
  // preamble.
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (!getPreambleFile(this).empty()) {
    using llvm::sys::FileStatus;
    llvm::sys::PathWithStatus CompleteFilePath(File);
    llvm::sys::PathWithStatus MainPath(OriginalSourceFile);
    if (const FileStatus *CompleteFileStatus = CompleteFilePath.getFileStatus())
      if (const FileStatus *MainStatus = MainPath.getFileStatus())
        if (CompleteFileStatus->getUniqueID() == MainStatus->getUniqueID() &&
            Line > 1)
          OverrideMainBuffer
            = getMainBufferWithPrecompiledPreamble(*CCInvocation, false, 
                                                   Line - 1);
  }

  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  StoredDiagnostics.insert(StoredDiagnostics.end(),
                           stored_diag_begin(),
                           stored_diag_afterDriver_begin());
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile, OverrideMainBuffer);
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PreprocessorOpts.ImplicitPCHInclude = getPreambleFile(this);
    PreprocessorOpts.DisablePCHValidation = true;
    
    OwnedBuffers.push_back(OverrideMainBuffer);
  } else {
    PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
    PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  }

  // Disable the preprocessing record if modules are not enabled.
  if (!Clang->getLangOpts().Modules)
    PreprocessorOpts.DetailedRecord = false;
  
  OwningPtr<SyntaxOnlyAction> Act;
  Act.reset(new SyntaxOnlyAction);
  if (Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0])) {
    if (OverrideMainBuffer) {
      std::string ModName = getPreambleFile(this);
      TranslateStoredDiagnostics(Clang->getModuleManager(), ModName,
                                 getSourceManager(), PreambleDiagnostics,
                                 StoredDiagnostics);
    }
    Act->Execute();
    Act->EndSourceFile();
  }

  checkAndSanitizeDiags(StoredDiagnostics, getSourceManager());
}

bool ASTUnit::Save(StringRef File) {
  // Write to a temporary file and later rename it to the actual file, to avoid
  // possible race conditions.
  SmallString<128> TempPath;
  TempPath = File;
  TempPath += "-%%%%%%%%";
  int fd;
  if (llvm::sys::fs::unique_file(TempPath.str(), fd, TempPath,
                                 /*makeAbsolute=*/false))
    return true;

  // FIXME: Can we somehow regenerate the stat cache here, or do we need to 
  // unconditionally create a stat cache when we parse the file?
  llvm::raw_fd_ostream Out(fd, /*shouldClose=*/true);

  serialize(Out);
  Out.close();
  if (Out.has_error()) {
    Out.clear_error();
    return true;
  }

  if (llvm::sys::fs::rename(TempPath.str(), File)) {
    bool exists;
    llvm::sys::fs::remove(TempPath.str(), exists);
    return true;
  }

  return false;
}

static bool serializeUnit(ASTWriter &Writer,
                          SmallVectorImpl<char> &Buffer,
                          Sema &S,
                          bool hasErrors,
                          raw_ostream &OS) {
  Writer.WriteAST(S, std::string(), 0, "", hasErrors);

  // Write the generated bitstream to "Out".
  if (!Buffer.empty())
    OS.write(Buffer.data(), Buffer.size());

  return false;
}

bool ASTUnit::serialize(raw_ostream &OS) {
  bool hasErrors = getDiagnostics().hasErrorOccurred();

  if (WriterData)
    return serializeUnit(WriterData->Writer, WriterData->Buffer,
                         getSema(), hasErrors, OS);

  SmallString<128> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ASTWriter Writer(Stream);
  return serializeUnit(Writer, Buffer, getSema(), hasErrors, OS);
}

typedef ContinuousRangeMap<unsigned, int, 2> SLocRemap;

static void TranslateSLoc(SourceLocation &L, SLocRemap &Remap) {
  unsigned Raw = L.getRawEncoding();
  const unsigned MacroBit = 1U << 31;
  L = SourceLocation::getFromRawEncoding((Raw & MacroBit) |
      ((Raw & ~MacroBit) + Remap.find(Raw & ~MacroBit)->second));
}

void ASTUnit::TranslateStoredDiagnostics(
                          ASTReader *MMan,
                          StringRef ModName,
                          SourceManager &SrcMgr,
                          const SmallVectorImpl<StoredDiagnostic> &Diags,
                          SmallVectorImpl<StoredDiagnostic> &Out) {
  // The stored diagnostic has the old source manager in it; update
  // the locations to refer into the new source manager. We also need to remap
  // all the locations to the new view. This includes the diag location, any
  // associated source ranges, and the source ranges of associated fix-its.
  // FIXME: There should be a cleaner way to do this.

  SmallVector<StoredDiagnostic, 4> Result;
  Result.reserve(Diags.size());
  assert(MMan && "Don't have a module manager");
  serialization::ModuleFile *Mod = MMan->ModuleMgr.lookup(ModName);
  assert(Mod && "Don't have preamble module");
  SLocRemap &Remap = Mod->SLocRemap;
  for (unsigned I = 0, N = Diags.size(); I != N; ++I) {
    // Rebuild the StoredDiagnostic.
    const StoredDiagnostic &SD = Diags[I];
    SourceLocation L = SD.getLocation();
    TranslateSLoc(L, Remap);
    FullSourceLoc Loc(L, SrcMgr);

    SmallVector<CharSourceRange, 4> Ranges;
    Ranges.reserve(SD.range_size());
    for (StoredDiagnostic::range_iterator I = SD.range_begin(),
                                          E = SD.range_end();
         I != E; ++I) {
      SourceLocation BL = I->getBegin();
      TranslateSLoc(BL, Remap);
      SourceLocation EL = I->getEnd();
      TranslateSLoc(EL, Remap);
      Ranges.push_back(CharSourceRange(SourceRange(BL, EL), I->isTokenRange()));
    }

    SmallVector<FixItHint, 2> FixIts;
    FixIts.reserve(SD.fixit_size());
    for (StoredDiagnostic::fixit_iterator I = SD.fixit_begin(),
                                          E = SD.fixit_end();
         I != E; ++I) {
      FixIts.push_back(FixItHint());
      FixItHint &FH = FixIts.back();
      FH.CodeToInsert = I->CodeToInsert;
      SourceLocation BL = I->RemoveRange.getBegin();
      TranslateSLoc(BL, Remap);
      SourceLocation EL = I->RemoveRange.getEnd();
      TranslateSLoc(EL, Remap);
      FH.RemoveRange = CharSourceRange(SourceRange(BL, EL),
                                       I->RemoveRange.isTokenRange());
    }

    Result.push_back(StoredDiagnostic(SD.getLevel(), SD.getID(), 
                                      SD.getMessage(), Loc, Ranges, FixIts));
  }
  Result.swap(Out);
}

static inline bool compLocDecl(std::pair<unsigned, Decl *> L,
                               std::pair<unsigned, Decl *> R) {
  return L.first < R.first;
}

void ASTUnit::addFileLevelDecl(Decl *D) {
  assert(D);
  
  // We only care about local declarations.
  if (D->isFromASTFile())
    return;

  SourceManager &SM = *SourceMgr;
  SourceLocation Loc = D->getLocation();
  if (Loc.isInvalid() || !SM.isLocalSourceLocation(Loc))
    return;

  // We only keep track of the file-level declarations of each file.
  if (!D->getLexicalDeclContext()->isFileContext())
    return;

  SourceLocation FileLoc = SM.getFileLoc(Loc);
  assert(SM.isLocalSourceLocation(FileLoc));
  FileID FID;
  unsigned Offset;
  llvm::tie(FID, Offset) = SM.getDecomposedLoc(FileLoc);
  if (FID.isInvalid())
    return;

  LocDeclsTy *&Decls = FileDecls[FID];
  if (!Decls)
    Decls = new LocDeclsTy();

  std::pair<unsigned, Decl *> LocDecl(Offset, D);

  if (Decls->empty() || Decls->back().first <= Offset) {
    Decls->push_back(LocDecl);
    return;
  }

  LocDeclsTy::iterator
    I = std::upper_bound(Decls->begin(), Decls->end(), LocDecl, compLocDecl);

  Decls->insert(I, LocDecl);
}

void ASTUnit::findFileRegionDecls(FileID File, unsigned Offset, unsigned Length,
                                  SmallVectorImpl<Decl *> &Decls) {
  if (File.isInvalid())
    return;

  if (SourceMgr->isLoadedFileID(File)) {
    assert(Ctx->getExternalSource() && "No external source!");
    return Ctx->getExternalSource()->FindFileRegionDecls(File, Offset, Length,
                                                         Decls);
  }

  FileDeclsTy::iterator I = FileDecls.find(File);
  if (I == FileDecls.end())
    return;

  LocDeclsTy &LocDecls = *I->second;
  if (LocDecls.empty())
    return;

  LocDeclsTy::iterator
    BeginIt = std::lower_bound(LocDecls.begin(), LocDecls.end(),
                               std::make_pair(Offset, (Decl*)0), compLocDecl);
  if (BeginIt != LocDecls.begin())
    --BeginIt;

  // If we are pointing at a top-level decl inside an objc container, we need
  // to backtrack until we find it otherwise we will fail to report that the
  // region overlaps with an objc container.
  while (BeginIt != LocDecls.begin() &&
         BeginIt->second->isTopLevelDeclInObjCContainer())
    --BeginIt;

  LocDeclsTy::iterator
    EndIt = std::upper_bound(LocDecls.begin(), LocDecls.end(),
                             std::make_pair(Offset+Length, (Decl*)0),
                             compLocDecl);
  if (EndIt != LocDecls.end())
    ++EndIt;
  
  for (LocDeclsTy::iterator DIt = BeginIt; DIt != EndIt; ++DIt)
    Decls.push_back(DIt->second);
}

SourceLocation ASTUnit::getLocation(const FileEntry *File,
                                    unsigned Line, unsigned Col) const {
  const SourceManager &SM = getSourceManager();
  SourceLocation Loc = SM.translateFileLineCol(File, Line, Col);
  return SM.getMacroArgExpandedLocation(Loc);
}

SourceLocation ASTUnit::getLocation(const FileEntry *File,
                                    unsigned Offset) const {
  const SourceManager &SM = getSourceManager();
  SourceLocation FileLoc = SM.translateFileLineCol(File, 1, 1);
  return SM.getMacroArgExpandedLocation(FileLoc.getLocWithOffset(Offset));
}

/// \brief If \arg Loc is a loaded location from the preamble, returns
/// the corresponding local location of the main file, otherwise it returns
/// \arg Loc.
SourceLocation ASTUnit::mapLocationFromPreamble(SourceLocation Loc) {
  FileID PreambleID;
  if (SourceMgr)
    PreambleID = SourceMgr->getPreambleFileID();

  if (Loc.isInvalid() || Preamble.empty() || PreambleID.isInvalid())
    return Loc;

  unsigned Offs;
  if (SourceMgr->isInFileID(Loc, PreambleID, &Offs) && Offs < Preamble.size()) {
    SourceLocation FileLoc
        = SourceMgr->getLocForStartOfFile(SourceMgr->getMainFileID());
    return FileLoc.getLocWithOffset(Offs);
  }

  return Loc;
}

/// \brief If \arg Loc is a local location of the main file but inside the
/// preamble chunk, returns the corresponding loaded location from the
/// preamble, otherwise it returns \arg Loc.
SourceLocation ASTUnit::mapLocationToPreamble(SourceLocation Loc) {
  FileID PreambleID;
  if (SourceMgr)
    PreambleID = SourceMgr->getPreambleFileID();

  if (Loc.isInvalid() || Preamble.empty() || PreambleID.isInvalid())
    return Loc;

  unsigned Offs;
  if (SourceMgr->isInFileID(Loc, SourceMgr->getMainFileID(), &Offs) &&
      Offs < Preamble.size()) {
    SourceLocation FileLoc = SourceMgr->getLocForStartOfFile(PreambleID);
    return FileLoc.getLocWithOffset(Offs);
  }

  return Loc;
}

bool ASTUnit::isInPreambleFileID(SourceLocation Loc) {
  FileID FID;
  if (SourceMgr)
    FID = SourceMgr->getPreambleFileID();
  
  if (Loc.isInvalid() || FID.isInvalid())
    return false;
  
  return SourceMgr->isInFileID(Loc, FID);
}

bool ASTUnit::isInMainFileID(SourceLocation Loc) {
  FileID FID;
  if (SourceMgr)
    FID = SourceMgr->getMainFileID();
  
  if (Loc.isInvalid() || FID.isInvalid())
    return false;
  
  return SourceMgr->isInFileID(Loc, FID);
}

SourceLocation ASTUnit::getEndOfPreambleFileID() {
  FileID FID;
  if (SourceMgr)
    FID = SourceMgr->getPreambleFileID();
  
  if (FID.isInvalid())
    return SourceLocation();

  return SourceMgr->getLocForEndOfFile(FID);
}

SourceLocation ASTUnit::getStartOfMainFileID() {
  FileID FID;
  if (SourceMgr)
    FID = SourceMgr->getMainFileID();
  
  if (FID.isInvalid())
    return SourceLocation();
  
  return SourceMgr->getLocForStartOfFile(FID);
}

std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
ASTUnit::getLocalPreprocessingEntities() const {
  if (isMainFileAST()) {
    serialization::ModuleFile &
      Mod = Reader->getModuleManager().getPrimaryModule();
    return Reader->getModulePreprocessedEntities(Mod);
  }

  if (PreprocessingRecord *PPRec = PP->getPreprocessingRecord())
    return std::make_pair(PPRec->local_begin(), PPRec->local_end());

  return std::make_pair(PreprocessingRecord::iterator(),
                        PreprocessingRecord::iterator());
}

bool ASTUnit::visitLocalTopLevelDecls(void *context, DeclVisitorFn Fn) {
  if (isMainFileAST()) {
    serialization::ModuleFile &
      Mod = Reader->getModuleManager().getPrimaryModule();
    ASTReader::ModuleDeclIterator MDI, MDE;
    llvm::tie(MDI, MDE) = Reader->getModuleFileLevelDecls(Mod);
    for (; MDI != MDE; ++MDI) {
      if (!Fn(context, *MDI))
        return false;
    }

    return true;
  }

  for (ASTUnit::top_level_iterator TL = top_level_begin(),
                                TLEnd = top_level_end();
         TL != TLEnd; ++TL) {
    if (!Fn(context, *TL))
      return false;
  }

  return true;
}

namespace {
struct PCHLocatorInfo {
  serialization::ModuleFile *Mod;
  PCHLocatorInfo() : Mod(0) {}
};
}

static bool PCHLocator(serialization::ModuleFile &M, void *UserData) {
  PCHLocatorInfo &Info = *static_cast<PCHLocatorInfo*>(UserData);
  switch (M.Kind) {
  case serialization::MK_Module:
    return true; // skip dependencies.
  case serialization::MK_PCH:
    Info.Mod = &M;
    return true; // found it.
  case serialization::MK_Preamble:
    return false; // look in dependencies.
  case serialization::MK_MainFile:
    return false; // look in dependencies.
  }

  return true;
}

const FileEntry *ASTUnit::getPCHFile() {
  if (!Reader)
    return 0;

  PCHLocatorInfo Info;
  Reader->getModuleManager().visit(PCHLocator, &Info);
  if (Info.Mod)
    return Info.Mod->File;

  return 0;
}

bool ASTUnit::isModuleFile() {
  return isMainFileAST() && !ASTFileLangOpts.CurrentModule.empty();
}

void ASTUnit::PreambleData::countLines() const {
  NumLines = 0;
  if (empty())
    return;

  for (std::vector<char>::const_iterator
         I = Buffer.begin(), E = Buffer.end(); I != E; ++I) {
    if (*I == '\n')
      ++NumLines;
  }
  if (Buffer.back() != '\n')
    ++NumLines;
}

#ifndef NDEBUG
ASTUnit::ConcurrencyState::ConcurrencyState() {
  Mutex = new llvm::sys::MutexImpl(/*recursive=*/true);
}

ASTUnit::ConcurrencyState::~ConcurrencyState() {
  delete static_cast<llvm::sys::MutexImpl *>(Mutex);
}

void ASTUnit::ConcurrencyState::start() {
  bool acquired = static_cast<llvm::sys::MutexImpl *>(Mutex)->tryacquire();
  assert(acquired && "Concurrent access to ASTUnit!");
}

void ASTUnit::ConcurrencyState::finish() {
  static_cast<llvm::sys::MutexImpl *>(Mutex)->release();
}

#else // NDEBUG

ASTUnit::ConcurrencyState::ConcurrencyState() {}
ASTUnit::ConcurrencyState::~ConcurrencyState() {}
void ASTUnit::ConcurrencyState::start() {}
void ASTUnit::ConcurrencyState::finish() {}

#endif
