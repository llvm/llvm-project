//===--- ASTUnit.cpp - ASTUnit utility --------------------------*- C++ -*-===//
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
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>

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
    SmallVector<std::string, 4> TemporaryFiles;

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

static void cleanupOnDiskMapAtExit();

typedef llvm::DenseMap<const ASTUnit *,
                       std::unique_ptr<OnDiskData>> OnDiskDataMap;
static OnDiskDataMap &getOnDiskDataMap() {
  static OnDiskDataMap M;
  static bool hasRegisteredAtExit = false;
  if (!hasRegisteredAtExit) {
    hasRegisteredAtExit = true;
    atexit(cleanupOnDiskMapAtExit);
  }
  return M;
}

static void cleanupOnDiskMapAtExit() {
  // Use the mutex because there can be an alive thread destroying an ASTUnit.
  llvm::MutexGuard Guard(getOnDiskMutex());
  for (const auto &I : getOnDiskDataMap()) {
    // We don't worry about freeing the memory associated with OnDiskDataMap.
    // All we care about is erasing stale files.
    I.second->Cleanup();
  }
}

static OnDiskData &getOnDiskData(const ASTUnit *AU) {
  // We require the mutex since we are modifying the structure of the
  // DenseMap.
  llvm::MutexGuard Guard(getOnDiskMutex());
  OnDiskDataMap &M = getOnDiskDataMap();
  auto &D = M[AU];
  if (!D)
    D = llvm::make_unique<OnDiskData>();
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
    M.erase(I);
  }
}

static void setPreambleFile(const ASTUnit *AU, StringRef preambleFile) {
  getOnDiskData(AU).PreambleFile = preambleFile;
}

static const std::string &getPreambleFile(const ASTUnit *AU) {
  return getOnDiskData(AU).PreambleFile;  
}

void OnDiskData::CleanTemporaryFiles() {
  for (StringRef File : TemporaryFiles)
    llvm::sys::fs::remove(File);
  TemporaryFiles.clear();
}

void OnDiskData::CleanPreambleFile() {
  if (!PreambleFile.empty()) {
    llvm::sys::fs::remove(PreambleFile);
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

  ASTWriterData() : Stream(Buffer), Writer(Stream, { }) { }
};

void ASTUnit::clearFileLevelDecls() {
  llvm::DeleteContainerSeconds(FileDecls);
}

void ASTUnit::CleanTemporaryFiles() {
  getOnDiskData(this).CleanTemporaryFiles();
}

void ASTUnit::addTemporaryFile(StringRef TempFile) {
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
static std::atomic<unsigned> ActiveASTUnitObjects;

ASTUnit::ASTUnit(bool _MainFileIsAST)
  : Reader(nullptr), HadModuleLoaderFatalFailure(false),
    OnlyLocalDecls(false), CaptureDiagnostics(false),
    MainFileIsAST(_MainFileIsAST), 
    TUKind(TU_Complete), WantTiming(getenv("LIBCLANG_TIMING")),
    OwnsRemappedFileBuffers(true),
    NumStoredDiagnosticsFromDriver(0),
    PreambleRebuildCounter(0),
    NumWarningsInPreamble(0),
    ShouldCacheCodeCompletionResults(false),
    IncludeBriefCommentsInCodeCompletion(false), UserFilesAreVolatile(false),
    CompletionCacheTopLevelHashValue(0),
    PreambleTopLevelHashValue(0),
    CurrentTopLevelHashValue(0),
    UnsafeToFree(false) { 
  if (getenv("LIBCLANG_OBJTRACKING"))
    fprintf(stderr, "+++ %u translation units\n", ++ActiveASTUnitObjects);
}

ASTUnit::~ASTUnit() {
  // If we loaded from an AST file, balance out the BeginSourceFile call.
  if (MainFileIsAST && getDiagnostics().getClient()) {
    getDiagnostics().getClient()->EndSourceFile();
  }

  clearFileLevelDecls();

  // Clean up the temporary files and the preamble file.
  removeOnDiskEntry(this);

  // Free the buffers associated with remapped files. We are required to
  // perform this operation here because we explicitly request that the
  // compiler instance *not* free these buffers for each invocation of the
  // parser.
  if (Invocation.get() && OwnsRemappedFileBuffers) {
    PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
    for (const auto &RB : PPOpts.RemappedFileBuffers)
      delete RB.second;
  }

  ClearCachedCompletionResults();  
  
  if (getenv("LIBCLANG_OBJTRACKING"))
    fprintf(stderr, "--- %u translation units\n", --ActiveASTUnitObjects);
}

void ASTUnit::setPreprocessor(Preprocessor *pp) { PP = pp; }

/// \brief Determine the set of code-completion contexts in which this 
/// declaration should be shown.
static unsigned getDeclShowContexts(const NamedDecl *ND,
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
      if (LangOpts.CPlusPlus11)
        IsNestedNameSpecifier = true;
    } else if (const RecordDecl *Record = dyn_cast<RecordDecl>(ND)) {
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
  CodeCompletionTUInfo CCTUInfo(CachedCompletionAllocator);
  TheSema->GatherGlobalCodeCompletions(*CachedCompletionAllocator,
                                       CCTUInfo, Results);
  
  // Translate global code completions into cached completions.
  llvm::DenseMap<CanQualType, unsigned> CompletionTypes;
  CodeCompletionContext CCContext(CodeCompletionContext::CCC_TopLevel);

  for (Result &R : Results) {
    switch (R.Kind) {
    case Result::RK_Declaration: {
      bool IsNestedNameSpecifier = false;
      CachedCodeCompletionResult CachedResult;
      CachedResult.Completion = R.CreateCodeCompletionString(
          *TheSema, CCContext, *CachedCompletionAllocator, CCTUInfo,
          IncludeBriefCommentsInCodeCompletion);
      CachedResult.ShowInContexts = getDeclShowContexts(
          R.Declaration, Ctx->getLangOpts(), IsNestedNameSpecifier);
      CachedResult.Priority = R.Priority;
      CachedResult.Kind = R.CursorKind;
      CachedResult.Availability = R.Availability;

      // Keep track of the type of this completion in an ASTContext-agnostic 
      // way.
      QualType UsageType = getDeclUsageType(*Ctx, R.Declaration);
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
      if (TheSema->Context.getLangOpts().CPlusPlus && IsNestedNameSpecifier &&
          !R.StartsNestedNameSpecifier) {
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

        if (isa<NamespaceDecl>(R.Declaration) ||
            isa<NamespaceAliasDecl>(R.Declaration))
          NNSContexts |= (1LL << CodeCompletionContext::CCC_Namespace);

        if (unsigned RemainingContexts 
                                = NNSContexts & ~CachedResult.ShowInContexts) {
          // If there any contexts where this completion can be a 
          // nested-name-specifier but isn't already an option, create a 
          // nested-name-specifier completion.
          R.StartsNestedNameSpecifier = true;
          CachedResult.Completion = R.CreateCodeCompletionString(
              *TheSema, CCContext, *CachedCompletionAllocator, CCTUInfo,
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
      CachedResult.Completion = R.CreateCodeCompletionString(
          *TheSema, CCContext, *CachedCompletionAllocator, CCTUInfo,
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

      CachedResult.Priority = R.Priority;
      CachedResult.Kind = R.CursorKind;
      CachedResult.Availability = R.Availability;
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
  CachedCompletionAllocator = nullptr;
}

namespace {

/// \brief Gathers information from ASTReader that will be used to initialize
/// a Preprocessor.
class ASTInfoCollector : public ASTReaderListener {
  Preprocessor &PP;
  ASTContext &Context;
  LangOptions &LangOpt;
  std::shared_ptr<TargetOptions> &TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> &Target;
  unsigned &Counter;

  bool InitializedLanguage;
public:
  ASTInfoCollector(Preprocessor &PP, ASTContext &Context, LangOptions &LangOpt,
                   std::shared_ptr<TargetOptions> &TargetOpts,
                   IntrusiveRefCntPtr<TargetInfo> &Target, unsigned &Counter)
      : PP(PP), Context(Context), LangOpt(LangOpt), TargetOpts(TargetOpts),
        Target(Target), Counter(Counter), InitializedLanguage(false) {}

  bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
                           bool AllowCompatibleDifferences) override {
    if (InitializedLanguage)
      return false;
    
    LangOpt = LangOpts;
    InitializedLanguage = true;
    
    updated();
    return false;
  }

  bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
                         bool AllowCompatibleDifferences) override {
    // If we've already initialized the target, don't do it again.
    if (Target)
      return false;

    this->TargetOpts = std::make_shared<TargetOptions>(TargetOpts);
    Target =
        TargetInfo::CreateTargetInfo(PP.getDiagnostics(), this->TargetOpts);

    updated();
    return false;
  }

  void ReadCounter(const serialization::ModuleFile &M,
                   unsigned Value) override {
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
    Target->adjust(LangOpt);

    // Initialize the preprocessor.
    PP.Initialize(*Target);

    // Initialize the ASTContext
    Context.InitBuiltinTypes(*Target);

    // We didn't have access to the comment options when the ASTContext was
    // constructed, so register them now.
    Context.getCommentCommandTraits().registerCommentOptions(
        LangOpt.CommentOpts);
  }
};

  /// \brief Diagnostic consumer that saves each diagnostic it is given.
class StoredDiagnosticConsumer : public DiagnosticConsumer {
  SmallVectorImpl<StoredDiagnostic> &StoredDiags;
  SourceManager *SourceMgr;

public:
  explicit StoredDiagnosticConsumer(
                          SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : StoredDiags(StoredDiags), SourceMgr(nullptr) {}

  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP = nullptr) override {
    if (PP)
      SourceMgr = &PP->getSourceManager();
  }

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override;
};

/// \brief RAII object that optionally captures diagnostics, if
/// there is no diagnostic client to capture them already.
class CaptureDroppedDiagnostics {
  DiagnosticsEngine &Diags;
  StoredDiagnosticConsumer Client;
  DiagnosticConsumer *PreviousClient;
  std::unique_ptr<DiagnosticConsumer> OwningPreviousClient;

public:
  CaptureDroppedDiagnostics(bool RequestCapture, DiagnosticsEngine &Diags,
                          SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : Diags(Diags), Client(StoredDiags), PreviousClient(nullptr)
  {
    if (RequestCapture || Diags.getClient() == nullptr) {
      OwningPreviousClient = Diags.takeClient();
      PreviousClient = Diags.getClient();
      Diags.setClient(&Client, false);
    }
  }

  ~CaptureDroppedDiagnostics() {
    if (Diags.getClient() == &Client)
      Diags.setClient(PreviousClient, !!OwningPreviousClient.release());
  }
};

} // anonymous namespace

void StoredDiagnosticConsumer::HandleDiagnostic(DiagnosticsEngine::Level Level,
                                              const Diagnostic &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(Level, Info);

  // Only record the diagnostic if it's part of the source manager we know
  // about. This effectively drops diagnostics from modules we're building.
  // FIXME: In the long run, ee don't want to drop source managers from modules.
  if (!Info.hasSourceManager() || &Info.getSourceManager() == SourceMgr)
    StoredDiags.emplace_back(Level, Info);
}

ASTMutationListener *ASTUnit::getASTMutationListener() {
  if (WriterData)
    return &WriterData->Writer;
  return nullptr;
}

ASTDeserializationListener *ASTUnit::getDeserializationListener() {
  if (WriterData)
    return &WriterData->Writer;
  return nullptr;
}

std::unique_ptr<llvm::MemoryBuffer>
ASTUnit::getBufferForFile(StringRef Filename, std::string *ErrorStr) {
  assert(FileMgr);
  auto Buffer = FileMgr->getBufferForFile(Filename);
  if (Buffer)
    return std::move(*Buffer);
  if (ErrorStr)
    *ErrorStr = Buffer.getError().message();
  return nullptr;
}

/// \brief Configure the diagnostics object for use with ASTUnit.
void ASTUnit::ConfigureDiags(IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                             ASTUnit &AST, bool CaptureDiagnostics) {
  assert(Diags.get() && "no DiagnosticsEngine was provided");
  if (CaptureDiagnostics)
    Diags->setClient(new StoredDiagnosticConsumer(AST.StoredDiagnostics));
}

std::unique_ptr<ASTUnit> ASTUnit::LoadFromASTFile(
    const std::string &Filename, const PCHContainerReader &PCHContainerRdr,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
    const FileSystemOptions &FileSystemOpts, bool UseDebugInfo,
    bool OnlyLocalDecls, ArrayRef<RemappedFile> RemappedFiles,
    bool CaptureDiagnostics, bool AllowPCHWithCompilerErrors,
    bool UserFilesAreVolatile) {
  std::unique_ptr<ASTUnit> AST(new ASTUnit(true));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(AST.get());
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.get());

  ConfigureDiags(Diags, *AST, CaptureDiagnostics);

  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->Diagnostics = Diags;
  IntrusiveRefCntPtr<vfs::FileSystem> VFS = vfs::getRealFileSystem();
  AST->FileMgr = new FileManager(FileSystemOpts, VFS);
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  AST->SourceMgr = new SourceManager(AST->getDiagnostics(),
                                     AST->getFileManager(),
                                     UserFilesAreVolatile);
  AST->HSOpts = new HeaderSearchOptions();
  AST->HSOpts->ModuleFormat = PCHContainerRdr.getFormat();
  AST->HeaderInfo.reset(new HeaderSearch(AST->HSOpts,
                                         AST->getSourceManager(),
                                         AST->getDiagnostics(),
                                         AST->ASTFileLangOpts,
                                         /*Target=*/nullptr));

  PreprocessorOptions *PPOpts = new PreprocessorOptions();

  for (const auto &RemappedFile : RemappedFiles)
    PPOpts->addRemappedFile(RemappedFile.first, RemappedFile.second);

  // Gather Info for preprocessor construction later on.

  HeaderSearch &HeaderInfo = *AST->HeaderInfo;
  unsigned Counter;

  AST->PP =
      new Preprocessor(PPOpts, AST->getDiagnostics(), AST->ASTFileLangOpts,
                       AST->getSourceManager(), HeaderInfo, *AST,
                       /*IILookup=*/nullptr,
                       /*OwnsHeaderSearch=*/false);
  Preprocessor &PP = *AST->PP;

  AST->Ctx = new ASTContext(AST->ASTFileLangOpts, AST->getSourceManager(),
                            PP.getIdentifierTable(), PP.getSelectorTable(),
                            PP.getBuiltinInfo());
  ASTContext &Context = *AST->Ctx;

  bool disableValid = false;
  if (::getenv("LIBCLANG_DISABLE_PCH_VALIDATION"))
    disableValid = true;
  AST->Reader = new ASTReader(PP, Context, PCHContainerRdr, { },
                              /*isysroot=*/"",
                              /*DisableValidation=*/disableValid,
                              AllowPCHWithCompilerErrors);

  AST->Reader->setListener(llvm::make_unique<ASTInfoCollector>(
      *AST->PP, Context, AST->ASTFileLangOpts, AST->TargetOpts, AST->Target,
      Counter));

  // Attach the AST reader to the AST context as an external AST
  // source, so that declarations will be deserialized from the
  // AST file as needed.
  // We need the external source to be set up before we read the AST, because
  // eagerly-deserialized declarations may use it.
  Context.setExternalSource(AST->Reader);

  switch (AST->Reader->ReadAST(Filename, serialization::MK_MainFile,
                          SourceLocation(), ASTReader::ARR_None)) {
  case ASTReader::Success:
    break;

  case ASTReader::Failure:
  case ASTReader::Missing:
  case ASTReader::OutOfDate:
  case ASTReader::VersionMismatch:
  case ASTReader::ConfigurationMismatch:
  case ASTReader::HadErrors:
    AST->getDiagnostics().Report(diag::err_fe_unable_to_load_pch);
    return nullptr;
  }

  AST->OriginalSourceFile = AST->Reader->getOriginalSourceFile();

  PP.setCounterValue(Counter);

  // Create an AST consumer, even though it isn't used.
  AST->Consumer.reset(new ASTConsumer);
  
  // Create a semantic analysis object and tell the AST reader about it.
  AST->TheSema.reset(new Sema(PP, Context, *AST->Consumer));
  AST->TheSema->Initialize();
  AST->Reader->InitializeSema(*AST->TheSema);

  // Tell the diagnostic client that we have started a source file.
  AST->getDiagnostics().getClient()->BeginSourceFile(Context.getLangOpts(),&PP);

  return AST;
}

namespace {

/// \brief Preprocessor callback class that updates a hash value with the names 
/// of all macros that have been defined by the translation unit.
class MacroDefinitionTrackerPPCallbacks : public PPCallbacks {
  unsigned &Hash;
  
public:
  explicit MacroDefinitionTrackerPPCallbacks(unsigned &Hash) : Hash(Hash) { }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
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
    if (EnumDecl *EnumD = dyn_cast<EnumDecl>(D)) {
      // For an unscoped enum include the enumerators in the hash since they
      // enter the top-level namespace.
      if (!EnumD->isScoped()) {
        for (const auto *EI : EnumD->enumerators()) {
          if (EI->getIdentifier())
            Hash = llvm::HashString(EI->getIdentifier()->getName(), Hash);
        }
      }
    }

    if (ND->getIdentifier())
      Hash = llvm::HashString(ND->getIdentifier()->getName(), Hash);
    else if (DeclarationName Name = ND->getDeclName()) {
      std::string NameStr = Name.getAsString();
      Hash = llvm::HashString(NameStr, Hash);
    }
    return;
  }

  if (ImportDecl *ImportD = dyn_cast<ImportDecl>(D)) {
    if (Module *Mod = ImportD->getImportedModule()) {
      std::string ModName = Mod->getFullModuleName();
      Hash = llvm::HashString(ModName, Hash);
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
      for (auto *I : NSD->decls())
        handleFileLevelDecl(I);
    }
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    for (Decl *TopLevelDecl : D)
      handleTopLevelDecl(TopLevelDecl);
    return true;
  }

  // We're not interested in "interesting" decls.
  void HandleInterestingDecl(DeclGroupRef) override {}

  void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) override {
    for (Decl *TopLevelDecl : D)
      handleTopLevelDecl(TopLevelDecl);
  }

  ASTMutationListener *GetASTMutationListener() override {
    return Unit.getASTMutationListener();
  }

  ASTDeserializationListener *GetASTDeserializationListener() override {
    return Unit.getDeserializationListener();
  }
};

class TopLevelDeclTrackerAction : public ASTFrontendAction {
public:
  ASTUnit &Unit;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    CI.getPreprocessor().addPPCallbacks(
        llvm::make_unique<MacroDefinitionTrackerPPCallbacks>(
                                           Unit.getCurrentTopLevelHashValue()));
    return llvm::make_unique<TopLevelDeclTrackerConsumer>(
        Unit, Unit.getCurrentTopLevelHashValue());
  }

public:
  TopLevelDeclTrackerAction(ASTUnit &_Unit) : Unit(_Unit) {}

  bool hasCodeCompletionSupport() const override { return false; }
  TranslationUnitKind getTranslationUnitKind() override {
    return Unit.getTranslationUnitKind(); 
  }
};

class PrecompilePreambleAction : public ASTFrontendAction {
  ASTUnit &Unit;
  bool HasEmittedPreamblePCH;

public:
  explicit PrecompilePreambleAction(ASTUnit &Unit)
      : Unit(Unit), HasEmittedPreamblePCH(false) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
  bool hasEmittedPreamblePCH() const { return HasEmittedPreamblePCH; }
  void setHasEmittedPreamblePCH() { HasEmittedPreamblePCH = true; }
  bool shouldEraseOutputFiles() override { return !hasEmittedPreamblePCH(); }

  bool hasCodeCompletionSupport() const override { return false; }
  bool hasASTFileSupport() const override { return false; }
  TranslationUnitKind getTranslationUnitKind() override { return TU_Prefix; }
};

class PrecompilePreambleConsumer : public PCHGenerator {
  ASTUnit &Unit;
  unsigned &Hash;
  std::vector<Decl *> TopLevelDecls;
  PrecompilePreambleAction *Action;
  raw_ostream *Out;

public:
  PrecompilePreambleConsumer(ASTUnit &Unit, PrecompilePreambleAction *Action,
                             const Preprocessor &PP, StringRef isysroot,
                             raw_ostream *Out)
      : PCHGenerator(PP, "", nullptr, isysroot, std::make_shared<PCHBuffer>(),
                     ArrayRef<llvm::IntrusiveRefCntPtr<ModuleFileExtension>>(),
                     /*AllowASTWithErrors=*/true),
        Unit(Unit), Hash(Unit.getCurrentTopLevelHashValue()), Action(Action),
        Out(Out) {
    Hash = 0;
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
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

  void HandleTranslationUnit(ASTContext &Ctx) override {
    PCHGenerator::HandleTranslationUnit(Ctx);
    if (hasEmittedPCH()) {
      // Write the generated bitstream to "Out".
      *Out << getPCH();
      // Make sure it hits disk now.
      Out->flush();
      // Free the buffer.
      llvm::SmallVector<char, 0> Empty;
      getPCH() = std::move(Empty);

      // Translate the top-level declarations we captured during
      // parsing into declaration IDs in the precompiled
      // preamble. This will allow us to deserialize those top-level
      // declarations when requested.
      for (Decl *D : TopLevelDecls) {
        // Invalid top-level decls may not have been serialized.
        if (D->isInvalidDecl())
          continue;
        Unit.addTopLevelDeclFromPreamble(getWriter().getDeclID(D));
      }

      Action->setHasEmittedPreamblePCH();
    }
  }
};

} // anonymous namespace

std::unique_ptr<ASTConsumer>
PrecompilePreambleAction::CreateASTConsumer(CompilerInstance &CI,
                                            StringRef InFile) {
  std::string Sysroot;
  std::string OutputFile;
  raw_ostream *OS = GeneratePCHAction::ComputeASTConsumerArguments(
      CI, InFile, Sysroot, OutputFile);
  if (!OS)
    return nullptr;

  if (!CI.getFrontendOpts().RelocatablePCH)
    Sysroot.clear();

  CI.getPreprocessor().addPPCallbacks(
      llvm::make_unique<MacroDefinitionTrackerPPCallbacks>(
                                           Unit.getCurrentTopLevelHashValue()));
  return llvm::make_unique<PrecompilePreambleConsumer>(
      Unit, this, CI.getPreprocessor(), Sysroot, OS);
}

static bool isNonDriverDiag(const StoredDiagnostic &StoredDiag) {
  return StoredDiag.getLocation().isValid();
}

static void
checkAndRemoveNonDriverDiags(SmallVectorImpl<StoredDiagnostic> &StoredDiags) {
  // Get rid of stored diagnostics except the ones from the driver which do not
  // have a source location.
  StoredDiags.erase(
      std::remove_if(StoredDiags.begin(), StoredDiags.end(), isNonDriverDiag),
      StoredDiags.end());
}

static void checkAndSanitizeDiags(SmallVectorImpl<StoredDiagnostic> &
                                                              StoredDiagnostics,
                                  SourceManager &SM) {
  // The stored diagnostic has the old source manager in it; update
  // the locations to refer into the new source manager. Since we've
  // been careful to make sure that the source manager's state
  // before and after are identical, so that we can reuse the source
  // location itself.
  for (StoredDiagnostic &SD : StoredDiagnostics) {
    if (SD.getLocation().isValid()) {
      FullSourceLoc Loc(SD.getLocation(), SM);
      SD.setLocation(Loc);
    }
  }
}

/// Parse the source file into a translation unit using the given compiler
/// invocation, replacing the current translation unit.
///
/// \returns True if a failure occurred that causes the ASTUnit not to
/// contain any translation-unit information, false otherwise.
bool ASTUnit::Parse(std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                    std::unique_ptr<llvm::MemoryBuffer> OverrideMainBuffer) {
  SavedMainFileBuffer.reset();

  if (!Invocation)
    return true;

  // Create the compiler instance to use for building the AST.
  std::unique_ptr<CompilerInstance> Clang(
      new CompilerInstance(PCHContainerOps));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  IntrusiveRefCntPtr<CompilerInvocation>
    CCInvocation(new CompilerInvocation(*Invocation));

  Clang->setInvocation(CCInvocation.get());
  OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
    
  // Set up diagnostics, capturing any diagnostics that would
  // otherwise be dropped.
  Clang->setDiagnostics(&getDiagnostics());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget())
    return true;

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().adjust(Clang->getLangOpts());
  
  assert(Clang->getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_LLVM_IR &&
         "IR inputs not support here!");

  // Configure the various subsystems.
  LangOpts = Clang->getInvocation().LangOpts;
  FileSystemOpts = Clang->getFileSystemOpts();
  if (!FileMgr) {
    Clang->createFileManager();
    FileMgr = &Clang->getFileManager();
  }
  SourceMgr = new SourceManager(getDiagnostics(), *FileMgr,
                                UserFilesAreVolatile);
  TheSema.reset();
  Ctx = nullptr;
  PP = nullptr;
  Reader = nullptr;

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
    PreprocessorOpts.addRemappedFile(OriginalSourceFile,
                                     OverrideMainBuffer.get());
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
    SavedMainFileBuffer = std::move(OverrideMainBuffer);
  }

  std::unique_ptr<TopLevelDeclTrackerAction> Act(
      new TopLevelDeclTrackerAction(*this));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<TopLevelDeclTrackerAction>
    ActCleanup(Act.get());

  if (!Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0]))
    goto error;

  if (SavedMainFileBuffer) {
    std::string ModName = getPreambleFile(this);
    TranslateStoredDiagnostics(getFileManager(), getSourceManager(),
                               PreambleDiagnostics, StoredDiagnostics);
  }

  if (!Act->Execute())
    goto error;

  transferASTDataFromCompilerInstance(*Clang);
  
  Act->EndSourceFile();

  FailedParseDiagnostics.clear();

  return false;

error:
  // Remove the overridden buffer we used for the preamble.
  SavedMainFileBuffer = nullptr;

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
  // FIXME: This is a hack so that we can override the preamble file during
  // crash-recovery testing, which is the only case where the preamble files
  // are not necessarily cleaned up.
  const char *TmpFile = ::getenv("CINDEXTEST_PREAMBLE_FILE");
  if (TmpFile)
    return TmpFile;

  SmallString<128> Path;
  llvm::sys::fs::createTemporaryFile("preamble", "pch", Path);

  return Path.str();
}

/// \brief Compute the preamble for the main file, providing the source buffer
/// that corresponds to the main file along with a pair (bytes, start-of-line)
/// that describes the preamble.
ASTUnit::ComputedPreamble
ASTUnit::ComputePreamble(CompilerInvocation &Invocation, unsigned MaxLines) {
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts = Invocation.getPreprocessorOpts();
  
  // Try to determine if the main file has been remapped, either from the 
  // command line (to another file) or directly through the compiler invocation
  // (to a memory buffer).
  llvm::MemoryBuffer *Buffer = nullptr;
  std::unique_ptr<llvm::MemoryBuffer> BufferOwner;
  std::string MainFilePath(FrontendOpts.Inputs[0].getFile());
  llvm::sys::fs::UniqueID MainFileID;
  if (!llvm::sys::fs::getUniqueID(MainFilePath, MainFileID)) {
    // Check whether there is a file-file remapping of the main file
    for (const auto &RF : PreprocessorOpts.RemappedFiles) {
      std::string MPath(RF.first);
      llvm::sys::fs::UniqueID MID;
      if (!llvm::sys::fs::getUniqueID(MPath, MID)) {
        if (MainFileID == MID) {
          // We found a remapping. Try to load the resulting, remapped source.
          BufferOwner = getBufferForFile(RF.second);
          if (!BufferOwner)
            return ComputedPreamble(nullptr, nullptr, 0, true);
        }
      }
    }
    
    // Check whether there is a file-buffer remapping. It supercedes the
    // file-file remapping.
    for (const auto &RB : PreprocessorOpts.RemappedFileBuffers) {
      std::string MPath(RB.first);
      llvm::sys::fs::UniqueID MID;
      if (!llvm::sys::fs::getUniqueID(MPath, MID)) {
        if (MainFileID == MID) {
          // We found a remapping.
          BufferOwner.reset();
          Buffer = const_cast<llvm::MemoryBuffer *>(RB.second);
        }
      }
    }
  }
  
  // If the main source file was not remapped, load it now.
  if (!Buffer && !BufferOwner) {
    BufferOwner = getBufferForFile(FrontendOpts.Inputs[0].getFile());
    if (!BufferOwner)
      return ComputedPreamble(nullptr, nullptr, 0, true);
  }

  if (!Buffer)
    Buffer = BufferOwner.get();
  auto Pre = Lexer::ComputePreamble(Buffer->getBuffer(),
                                    *Invocation.getLangOpts(), MaxLines);
  return ComputedPreamble(Buffer, std::move(BufferOwner), Pre.first,
                          Pre.second);
}

ASTUnit::PreambleFileHash
ASTUnit::PreambleFileHash::createForFile(off_t Size, time_t ModTime) {
  PreambleFileHash Result;
  Result.Size = Size;
  Result.ModTime = ModTime;
  memset(Result.MD5, 0, sizeof(Result.MD5));
  return Result;
}

ASTUnit::PreambleFileHash ASTUnit::PreambleFileHash::createForMemoryBuffer(
    const llvm::MemoryBuffer *Buffer) {
  PreambleFileHash Result;
  Result.Size = Buffer->getBufferSize();
  Result.ModTime = 0;

  llvm::MD5 MD5Ctx;
  MD5Ctx.update(Buffer->getBuffer().data());
  MD5Ctx.final(Result.MD5);

  return Result;
}

namespace clang {
bool operator==(const ASTUnit::PreambleFileHash &LHS,
                const ASTUnit::PreambleFileHash &RHS) {
  return LHS.Size == RHS.Size && LHS.ModTime == RHS.ModTime &&
         memcmp(LHS.MD5, RHS.MD5, sizeof(LHS.MD5)) == 0;
}
} // namespace clang

static std::pair<unsigned, unsigned>
makeStandaloneRange(CharSourceRange Range, const SourceManager &SM,
                    const LangOptions &LangOpts) {
  CharSourceRange FileRange = Lexer::makeFileCharRange(Range, SM, LangOpts);
  unsigned Offset = SM.getFileOffset(FileRange.getBegin());
  unsigned EndOffset = SM.getFileOffset(FileRange.getEnd());
  return std::make_pair(Offset, EndOffset);
}

static ASTUnit::StandaloneFixIt makeStandaloneFixIt(const SourceManager &SM,
                                                    const LangOptions &LangOpts,
                                                    const FixItHint &InFix) {
  ASTUnit::StandaloneFixIt OutFix;
  OutFix.RemoveRange = makeStandaloneRange(InFix.RemoveRange, SM, LangOpts);
  OutFix.InsertFromRange = makeStandaloneRange(InFix.InsertFromRange, SM,
                                               LangOpts);
  OutFix.CodeToInsert = InFix.CodeToInsert;
  OutFix.BeforePreviousInsertions = InFix.BeforePreviousInsertions;
  return OutFix;
}

static ASTUnit::StandaloneDiagnostic
makeStandaloneDiagnostic(const LangOptions &LangOpts,
                         const StoredDiagnostic &InDiag) {
  ASTUnit::StandaloneDiagnostic OutDiag;
  OutDiag.ID = InDiag.getID();
  OutDiag.Level = InDiag.getLevel();
  OutDiag.Message = InDiag.getMessage();
  OutDiag.LocOffset = 0;
  if (InDiag.getLocation().isInvalid())
    return OutDiag;
  const SourceManager &SM = InDiag.getLocation().getManager();
  SourceLocation FileLoc = SM.getFileLoc(InDiag.getLocation());
  OutDiag.Filename = SM.getFilename(FileLoc);
  if (OutDiag.Filename.empty())
    return OutDiag;
  OutDiag.LocOffset = SM.getFileOffset(FileLoc);
  for (const CharSourceRange &Range : InDiag.getRanges())
    OutDiag.Ranges.push_back(makeStandaloneRange(Range, SM, LangOpts));
  for (const FixItHint &FixIt : InDiag.getFixIts())
    OutDiag.FixIts.push_back(makeStandaloneFixIt(SM, LangOpts, FixIt));

  return OutDiag;
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
std::unique_ptr<llvm::MemoryBuffer>
ASTUnit::getMainBufferWithPrecompiledPreamble(
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    const CompilerInvocation &PreambleInvocationIn, bool AllowRebuild,
    unsigned MaxLines) {

  IntrusiveRefCntPtr<CompilerInvocation>
    PreambleInvocation(new CompilerInvocation(PreambleInvocationIn));
  FrontendOptions &FrontendOpts = PreambleInvocation->getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts
    = PreambleInvocation->getPreprocessorOpts();

  ComputedPreamble NewPreamble = ComputePreamble(*PreambleInvocation, MaxLines);

  if (!NewPreamble.Size) {
    // We couldn't find a preamble in the main source. Clear out the current
    // preamble, if we have one. It's obviously no good any more.
    Preamble.clear();
    erasePreambleFile(this);

    // The next time we actually see a preamble, precompile it.
    PreambleRebuildCounter = 1;
    return nullptr;
  }
  
  if (!Preamble.empty()) {
    // We've previously computed a preamble. Check whether we have the same
    // preamble now that we did before, and that there's enough space in
    // the main-file buffer within the precompiled preamble to fit the
    // new main file.
    if (Preamble.size() == NewPreamble.Size &&
        PreambleEndsAtStartOfLine == NewPreamble.PreambleEndsAtStartOfLine &&
        memcmp(Preamble.getBufferStart(), NewPreamble.Buffer->getBufferStart(),
               NewPreamble.Size) == 0) {
      // The preamble has not changed. We may be able to re-use the precompiled
      // preamble.

      // Check that none of the files used by the preamble have changed.
      bool AnyFileChanged = false;
          
      // First, make a record of those files that have been overridden via
      // remapping or unsaved_files.
      llvm::StringMap<PreambleFileHash> OverriddenFiles;
      for (const auto &R : PreprocessorOpts.RemappedFiles) {
        if (AnyFileChanged)
          break;

        vfs::Status Status;
        if (FileMgr->getNoncachedStatValue(R.second, Status)) {
          // If we can't stat the file we're remapping to, assume that something
          // horrible happened.
          AnyFileChanged = true;
          break;
        }

        OverriddenFiles[R.first] = PreambleFileHash::createForFile(
            Status.getSize(), Status.getLastModificationTime().toEpochTime());
      }

      for (const auto &RB : PreprocessorOpts.RemappedFileBuffers) {
        if (AnyFileChanged)
          break;
        OverriddenFiles[RB.first] =
            PreambleFileHash::createForMemoryBuffer(RB.second);
      }
       
      // Check whether anything has changed.
      for (llvm::StringMap<PreambleFileHash>::iterator 
             F = FilesInPreamble.begin(), FEnd = FilesInPreamble.end();
           !AnyFileChanged && F != FEnd; 
           ++F) {
        llvm::StringMap<PreambleFileHash>::iterator Overridden
          = OverriddenFiles.find(F->first());
        if (Overridden != OverriddenFiles.end()) {
          // This file was remapped; check whether the newly-mapped file 
          // matches up with the previous mapping.
          if (Overridden->second != F->second)
            AnyFileChanged = true;
          continue;
        }
        
        // The file was not remapped; check whether it has changed on disk.
        vfs::Status Status;
        if (FileMgr->getNoncachedStatValue(F->first(), Status)) {
          // If we can't stat the file, assume that something horrible happened.
          AnyFileChanged = true;
        } else if (Status.getSize() != uint64_t(F->second.Size) ||
                   Status.getLastModificationTime().toEpochTime() !=
                       uint64_t(F->second.ModTime))
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

        return llvm::MemoryBuffer::getMemBufferCopy(
            NewPreamble.Buffer->getBuffer(), FrontendOpts.Inputs[0].getFile());
      }
    }

    // If we aren't allowed to rebuild the precompiled preamble, just
    // return now.
    if (!AllowRebuild)
      return nullptr;

    // We can't reuse the previously-computed preamble. Build a new one.
    Preamble.clear();
    PreambleDiagnostics.clear();
    erasePreambleFile(this);
    PreambleRebuildCounter = 1;
  } else if (!AllowRebuild) {
    // We aren't allowed to rebuild the precompiled preamble; just
    // return now.
    return nullptr;
  }

  // If the preamble rebuild counter > 1, it's because we previously
  // failed to build a preamble and we're not yet ready to try
  // again. Decrement the counter and return a failure.
  if (PreambleRebuildCounter > 1) {
    --PreambleRebuildCounter;
    return nullptr;
  }

  // Create a temporary file for the precompiled preamble. In rare 
  // circumstances, this can fail.
  std::string PreamblePCHPath = GetPreamblePCHPath();
  if (PreamblePCHPath.empty()) {
    // Try again next time.
    PreambleRebuildCounter = 1;
    return nullptr;
  }
  
  // We did not previously compute a preamble, or it can't be reused anyway.
  SimpleTimer PreambleTimer(WantTiming);
  PreambleTimer.setOutput("Precompiling preamble");

  // Save the preamble text for later; we'll need to compare against it for
  // subsequent reparses.
  StringRef MainFilename = FrontendOpts.Inputs[0].getFile();
  Preamble.assign(FileMgr->getFile(MainFilename),
                  NewPreamble.Buffer->getBufferStart(),
                  NewPreamble.Buffer->getBufferStart() + NewPreamble.Size);
  PreambleEndsAtStartOfLine = NewPreamble.PreambleEndsAtStartOfLine;

  PreambleBuffer = llvm::MemoryBuffer::getMemBufferCopy(
      NewPreamble.Buffer->getBuffer().slice(0, Preamble.size()), MainFilename);

  // Remap the main source file to the preamble buffer.
  StringRef MainFilePath = FrontendOpts.Inputs[0].getFile();
  PreprocessorOpts.addRemappedFile(MainFilePath, PreambleBuffer.get());

  // Tell the compiler invocation to generate a temporary precompiled header.
  FrontendOpts.ProgramAction = frontend::GeneratePCH;
  // FIXME: Generate the precompiled header into memory?
  FrontendOpts.OutputFile = PreamblePCHPath;
  PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
  PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  
  // Create the compiler instance to use for building the precompiled preamble.
  std::unique_ptr<CompilerInstance> Clang(
      new CompilerInstance(PCHContainerOps));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  Clang->setInvocation(&*PreambleInvocation);
  OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
  
  // Set up diagnostics, capturing all of the diagnostics produced.
  Clang->setDiagnostics(&getDiagnostics());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget()) {
    llvm::sys::fs::remove(FrontendOpts.OutputFile);
    Preamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.RemappedFileBuffers.pop_back();
    return nullptr;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().adjust(Clang->getLangOpts());
  
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
  PreambleDiagnostics.clear();

  IntrusiveRefCntPtr<vfs::FileSystem> VFS =
      createVFSFromCompilerInvocation(Clang->getInvocation(), getDiagnostics());
  if (!VFS)
    return nullptr;

  // Create a file manager object to provide access to and cache the filesystem.
  Clang->setFileManager(new FileManager(Clang->getFileSystemOpts(), VFS));
  
  // Create the source manager.
  Clang->setSourceManager(new SourceManager(getDiagnostics(),
                                            Clang->getFileManager()));

  auto PreambleDepCollector = std::make_shared<DependencyCollector>();
  Clang->addDependencyCollector(PreambleDepCollector);

  std::unique_ptr<PrecompilePreambleAction> Act;
  Act.reset(new PrecompilePreambleAction(*this));
  if (!Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0])) {
    llvm::sys::fs::remove(FrontendOpts.OutputFile);
    Preamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.RemappedFileBuffers.pop_back();
    return nullptr;
  }
  
  Act->Execute();

  // Transfer any diagnostics generated when parsing the preamble into the set
  // of preamble diagnostics.
  for (stored_diag_iterator I = stored_diag_afterDriver_begin(),
                            E = stored_diag_end();
       I != E; ++I)
    PreambleDiagnostics.push_back(
        makeStandaloneDiagnostic(Clang->getLangOpts(), *I));

  Act->EndSourceFile();

  checkAndRemoveNonDriverDiags(StoredDiagnostics);

  if (!Act->hasEmittedPreamblePCH()) {
    // The preamble PCH failed (e.g. there was a module loading fatal error),
    // so no precompiled header was generated. Forget that we even tried.
    // FIXME: Should we leave a note for ourselves to try again?
    llvm::sys::fs::remove(FrontendOpts.OutputFile);
    Preamble.clear();
    TopLevelDeclsInPreamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.RemappedFileBuffers.pop_back();
    return nullptr;
  }
  
  // Keep track of the preamble we precompiled.
  setPreambleFile(this, FrontendOpts.OutputFile);
  NumWarningsInPreamble = getDiagnostics().getNumWarnings();
  
  // Keep track of all of the files that the source manager knows about,
  // so we can verify whether they have changed or not.
  FilesInPreamble.clear();
  SourceManager &SourceMgr = Clang->getSourceManager();
  for (auto &Filename : PreambleDepCollector->getDependencies()) {
    const FileEntry *File = Clang->getFileManager().getFile(Filename);
    if (!File || File == SourceMgr.getFileEntryForID(SourceMgr.getMainFileID()))
      continue;
    if (time_t ModTime = File->getModificationTime()) {
      FilesInPreamble[File->getName()] = PreambleFileHash::createForFile(
          File->getSize(), ModTime);
    } else {
      llvm::MemoryBuffer *Buffer = SourceMgr.getMemoryBufferForFile(File);
      FilesInPreamble[File->getName()] =
          PreambleFileHash::createForMemoryBuffer(Buffer);
    }
  }

  PreambleRebuildCounter = 1;
  PreprocessorOpts.RemappedFileBuffers.pop_back();

  // If the hash of top-level entities differs from the hash of the top-level
  // entities the last time we rebuilt the preamble, clear out the completion
  // cache.
  if (CurrentTopLevelHashValue != PreambleTopLevelHashValue) {
    CompletionCacheTopLevelHashValue = 0;
    PreambleTopLevelHashValue = CurrentTopLevelHashValue;
  }

  return llvm::MemoryBuffer::getMemBufferCopy(NewPreamble.Buffer->getBuffer(),
                                              MainFilename);
}

void ASTUnit::RealizeTopLevelDeclsFromPreamble() {
  std::vector<Decl *> Resolved;
  Resolved.reserve(TopLevelDeclsInPreamble.size());
  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (serialization::DeclID TopLevelDecl : TopLevelDeclsInPreamble) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    if (Decl *D = Source.GetExternalDecl(TopLevelDecl))
      Resolved.push_back(D);
  }
  TopLevelDeclsInPreamble.clear();
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());
}

void ASTUnit::transferASTDataFromCompilerInstance(CompilerInstance &CI) {
  // Steal the created target, context, and preprocessor if they have been
  // created.
  assert(CI.hasInvocation() && "missing invocation");
  LangOpts = CI.getInvocation().LangOpts;
  TheSema = CI.takeSema();
  Consumer = CI.takeASTConsumer();
  if (CI.hasASTContext())
    Ctx = &CI.getASTContext();
  if (CI.hasPreprocessor())
    PP = &CI.getPreprocessor();
  CI.setSourceManager(nullptr);
  CI.setFileManager(nullptr);
  if (CI.hasTarget())
    Target = &CI.getTarget();
  Reader = CI.getModuleManager();
  HadModuleLoaderFatalFailure = CI.hadModuleLoaderFatalFailure();
}

StringRef ASTUnit::getMainFileName() const {
  if (Invocation && !Invocation->getFrontendOpts().Inputs.empty()) {
    const FrontendInputFile &Input = Invocation->getFrontendOpts().Inputs[0];
    if (Input.isFile())
      return Input.getFile();
    else
      return Input.getBuffer()->getBufferIdentifier();
  }

  if (SourceMgr) {
    if (const FileEntry *
          FE = SourceMgr->getFileEntryForID(SourceMgr->getMainFileID()))
      return FE->getName();
  }

  return StringRef();
}

StringRef ASTUnit::getASTFileName() const {
  if (!isMainFileAST())
    return StringRef();

  serialization::ModuleFile &
    Mod = Reader->getModuleManager().getPrimaryModule();
  return Mod.FileName;
}

ASTUnit *ASTUnit::create(CompilerInvocation *CI,
                         IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                         bool CaptureDiagnostics,
                         bool UserFilesAreVolatile) {
  std::unique_ptr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  AST->Invocation = CI;
  AST->FileSystemOpts = CI->getFileSystemOpts();
  IntrusiveRefCntPtr<vfs::FileSystem> VFS =
      createVFSFromCompilerInvocation(*CI, *Diags);
  if (!VFS)
    return nullptr;
  AST->FileMgr = new FileManager(AST->FileSystemOpts, VFS);
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  AST->SourceMgr = new SourceManager(AST->getDiagnostics(), *AST->FileMgr,
                                     UserFilesAreVolatile);

  return AST.release();
}

ASTUnit *ASTUnit::LoadFromCompilerInvocationAction(
    CompilerInvocation *CI,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags, FrontendAction *Action,
    ASTUnit *Unit, bool Persistent, StringRef ResourceFilesPath,
    bool OnlyLocalDecls, bool CaptureDiagnostics,
    unsigned PrecompilePreambleAfterNParses, bool CacheCodeCompletionResults,
    bool IncludeBriefCommentsInCodeCompletion, bool UserFilesAreVolatile,
    std::unique_ptr<ASTUnit> *ErrAST) {
  assert(CI && "A CompilerInvocation is required");

  std::unique_ptr<ASTUnit> OwnAST;
  ASTUnit *AST = Unit;
  if (!AST) {
    // Create the AST unit.
    OwnAST.reset(create(CI, Diags, CaptureDiagnostics, UserFilesAreVolatile));
    AST = OwnAST.get();
    if (!AST)
      return nullptr;
  }
  
  if (!ResourceFilesPath.empty()) {
    // Override the resources path.
    CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;
  }
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  if (PrecompilePreambleAfterNParses > 0)
    AST->PreambleRebuildCounter = PrecompilePreambleAfterNParses;
  AST->TUKind = Action ? Action->getTranslationUnitKind() : TU_Complete;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->IncludeBriefCommentsInCodeCompletion
    = IncludeBriefCommentsInCodeCompletion;

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(OwnAST.get());
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.get());

  // We'll manage file buffers ourselves.
  CI->getPreprocessorOpts().RetainRemappedFileBuffers = true;
  CI->getFrontendOpts().DisableFree = false;
  ProcessWarningOptions(AST->getDiagnostics(), CI->getDiagnosticOpts());

  // Create the compiler instance to use for building the AST.
  std::unique_ptr<CompilerInstance> Clang(
      new CompilerInstance(PCHContainerOps));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  Clang->setInvocation(CI);
  AST->OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
    
  // Set up diagnostics, capturing any diagnostics that would
  // otherwise be dropped.
  Clang->setDiagnostics(&AST->getDiagnostics());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget())
    return nullptr;

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().adjust(Clang->getLangOpts());
  
  assert(Clang->getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang->getFrontendOpts().Inputs[0].getKind() != IK_LLVM_IR &&
         "IR inputs not supported here!");

  // Configure the various subsystems.
  AST->TheSema.reset();
  AST->Ctx = nullptr;
  AST->PP = nullptr;
  AST->Reader = nullptr;

  // Create a file manager object to provide access to and cache the filesystem.
  Clang->setFileManager(&AST->getFileManager());
  
  // Create the source manager.
  Clang->setSourceManager(&AST->getSourceManager());

  FrontendAction *Act = Action;

  std::unique_ptr<TopLevelDeclTrackerAction> TrackerAct;
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

    return nullptr;
  }

  if (Persistent && !TrackerAct) {
    Clang->getPreprocessor().addPPCallbacks(
        llvm::make_unique<MacroDefinitionTrackerPPCallbacks>(
                                           AST->getCurrentTopLevelHashValue()));
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    if (Clang->hasASTConsumer())
      Consumers.push_back(Clang->takeASTConsumer());
    Consumers.push_back(llvm::make_unique<TopLevelDeclTrackerConsumer>(
        *AST, AST->getCurrentTopLevelHashValue()));
    Clang->setASTConsumer(
        llvm::make_unique<MultiplexConsumer>(std::move(Consumers)));
  }
  if (!Act->Execute()) {
    AST->transferASTDataFromCompilerInstance(*Clang);
    if (OwnAST && ErrAST)
      ErrAST->swap(OwnAST);

    return nullptr;
  }

  // Steal the created target, context, and preprocessor.
  AST->transferASTDataFromCompilerInstance(*Clang);
  
  Act->EndSourceFile();

  if (OwnAST)
    return OwnAST.release();
  else
    return AST;
}

bool ASTUnit::LoadFromCompilerInvocation(
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    unsigned PrecompilePreambleAfterNParses) {
  if (!Invocation)
    return true;
  
  // We'll manage file buffers ourselves.
  Invocation->getPreprocessorOpts().RetainRemappedFileBuffers = true;
  Invocation->getFrontendOpts().DisableFree = false;
  ProcessWarningOptions(getDiagnostics(), Invocation->getDiagnosticOpts());

  std::unique_ptr<llvm::MemoryBuffer> OverrideMainBuffer;
  if (PrecompilePreambleAfterNParses > 0) {
    PreambleRebuildCounter = PrecompilePreambleAfterNParses;
    OverrideMainBuffer =
        getMainBufferWithPrecompiledPreamble(PCHContainerOps, *Invocation);
  }
  
  SimpleTimer ParsingTimer(WantTiming);
  ParsingTimer.setOutput("Parsing " + getMainFileName());
  
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<llvm::MemoryBuffer>
    MemBufferCleanup(OverrideMainBuffer.get());

  return Parse(PCHContainerOps, std::move(OverrideMainBuffer));
}

std::unique_ptr<ASTUnit> ASTUnit::LoadFromCompilerInvocation(
    CompilerInvocation *CI,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags, FileManager *FileMgr,
    bool OnlyLocalDecls, bool CaptureDiagnostics,
    unsigned PrecompilePreambleAfterNParses, TranslationUnitKind TUKind,
    bool CacheCodeCompletionResults, bool IncludeBriefCommentsInCodeCompletion,
    bool UserFilesAreVolatile) {
  // Create the AST unit.
  std::unique_ptr<ASTUnit> AST(new ASTUnit(false));
  ConfigureDiags(Diags, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->TUKind = TUKind;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->IncludeBriefCommentsInCodeCompletion
    = IncludeBriefCommentsInCodeCompletion;
  AST->Invocation = CI;
  AST->FileSystemOpts = FileMgr->getFileSystemOpts();
  AST->FileMgr = FileMgr;
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  
  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(AST.get());
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.get());

  if (AST->LoadFromCompilerInvocation(PCHContainerOps,
                                      PrecompilePreambleAfterNParses))
    return nullptr;
  return AST;
}

ASTUnit *ASTUnit::LoadFromCommandLine(
    const char **ArgBegin, const char **ArgEnd,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags, StringRef ResourceFilesPath,
    bool OnlyLocalDecls, bool CaptureDiagnostics,
    ArrayRef<RemappedFile> RemappedFiles, bool RemappedFilesKeepOriginalName,
    unsigned PrecompilePreambleAfterNParses, TranslationUnitKind TUKind,
    bool CacheCodeCompletionResults, bool IncludeBriefCommentsInCodeCompletion,
    bool AllowPCHWithCompilerErrors, bool SkipFunctionBodies,
    bool UserFilesAreVolatile, bool ForSerialization,
    llvm::Optional<StringRef> ModuleFormat, std::unique_ptr<ASTUnit> *ErrAST) {
  assert(Diags.get() && "no DiagnosticsEngine was provided");

  SmallVector<StoredDiagnostic, 4> StoredDiagnostics;
  
  IntrusiveRefCntPtr<CompilerInvocation> CI;

  {

    CaptureDroppedDiagnostics Capture(CaptureDiagnostics, *Diags, 
                                      StoredDiagnostics);

    CI = clang::createInvocationFromCommandLine(
                                           llvm::makeArrayRef(ArgBegin, ArgEnd),
                                           Diags);
    if (!CI)
      return nullptr;
  }

  // Override any files that need remapping
  for (const auto &RemappedFile : RemappedFiles) {
    CI->getPreprocessorOpts().addRemappedFile(RemappedFile.first,
                                              RemappedFile.second);
  }
  PreprocessorOptions &PPOpts = CI->getPreprocessorOpts();
  PPOpts.RemappedFilesKeepOriginalName = RemappedFilesKeepOriginalName;
  PPOpts.AllowPCHWithCompilerErrors = AllowPCHWithCompilerErrors;
  
  // Override the resources path.
  CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;

  CI->getFrontendOpts().SkipFunctionBodies = SkipFunctionBodies;

  if (ModuleFormat)
    CI->getHeaderSearchOpts().ModuleFormat = ModuleFormat.getValue();

  // Create the AST unit.
  std::unique_ptr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  AST->FileSystemOpts = CI->getFileSystemOpts();
  IntrusiveRefCntPtr<vfs::FileSystem> VFS =
      createVFSFromCompilerInvocation(*CI, *Diags);
  if (!VFS)
    return nullptr;
  AST->FileMgr = new FileManager(AST->FileSystemOpts, VFS);
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
  // Zero out now to ease cleanup during crash recovery.
  CI = nullptr;
  Diags = nullptr;

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit>
    ASTUnitCleanup(AST.get());

  if (AST->LoadFromCompilerInvocation(PCHContainerOps,
                                      PrecompilePreambleAfterNParses)) {
    // Some error occurred, if caller wants to examine diagnostics, pass it the
    // ASTUnit.
    if (ErrAST) {
      AST->StoredDiagnostics.swap(AST->FailedParseDiagnostics);
      ErrAST->swap(AST);
    }
    return nullptr;
  }

  return AST.release();
}

bool ASTUnit::Reparse(std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                      ArrayRef<RemappedFile> RemappedFiles) {
  if (!Invocation)
    return true;

  clearFileLevelDecls();
  
  SimpleTimer ParsingTimer(WantTiming);
  ParsingTimer.setOutput("Reparsing " + getMainFileName());

  // Remap files.
  PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
  for (const auto &RB : PPOpts.RemappedFileBuffers)
    delete RB.second;

  Invocation->getPreprocessorOpts().clearRemappedFiles();
  for (const auto &RemappedFile : RemappedFiles) {
    Invocation->getPreprocessorOpts().addRemappedFile(RemappedFile.first,
                                                      RemappedFile.second);
  }

  // If we have a preamble file lying around, or if we might try to
  // build a precompiled preamble, do so now.
  std::unique_ptr<llvm::MemoryBuffer> OverrideMainBuffer;
  if (!getPreambleFile(this).empty() || PreambleRebuildCounter > 0)
    OverrideMainBuffer =
        getMainBufferWithPrecompiledPreamble(PCHContainerOps, *Invocation);

  // Clear out the diagnostics state.
  FileMgr.reset();
  getDiagnostics().Reset();
  ProcessWarningOptions(getDiagnostics(), Invocation->getDiagnosticOpts());
  if (OverrideMainBuffer)
    getDiagnostics().setNumWarnings(NumWarningsInPreamble);

  // Parse the sources
  bool Result = Parse(PCHContainerOps, std::move(OverrideMainBuffer));

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

    void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                    CodeCompletionResult *Results,
                                    unsigned NumResults) override;

    void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                   OverloadCandidate *Candidates,
                                   unsigned NumCandidates) override {
      Next.ProcessOverloadCandidates(S, CurrentArg, Candidates, NumCandidates);
    }

    CodeCompletionAllocator &getAllocator() override {
      return Next.getAllocator();
    }

    CodeCompletionTUInfo &getCodeCompletionTUInfo() override {
      return Next.getCodeCompletionTUInfo();
    }
  };
} // anonymous namespace

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

void ASTUnit::CodeComplete(
    StringRef File, unsigned Line, unsigned Column,
    ArrayRef<RemappedFile> RemappedFiles, bool IncludeMacros,
    bool IncludeCodePatterns, bool IncludeBriefComments,
    CodeCompleteConsumer &Consumer,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    DiagnosticsEngine &Diag, LangOptions &LangOpts, SourceManager &SourceMgr,
    FileManager &FileMgr, SmallVectorImpl<StoredDiagnostic> &StoredDiagnostics,
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

  // Spell-checking and warnings are wasteful during code-completion.
  LangOpts.SpellChecking = false;
  CCInvocation->getDiagnosticOpts().IgnoreWarnings = true;

  std::unique_ptr<CompilerInstance> Clang(
      new CompilerInstance(PCHContainerOps));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance>
    CICleanup(Clang.get());

  Clang->setInvocation(&*CCInvocation);
  OriginalSourceFile = Clang->getFrontendOpts().Inputs[0].getFile();
    
  // Set up diagnostics, capturing any diagnostics produced.
  Clang->setDiagnostics(&Diag);
  CaptureDroppedDiagnostics Capture(true, 
                                    Clang->getDiagnostics(), 
                                    StoredDiagnostics);
  ProcessWarningOptions(Diag, CCInvocation->getDiagnosticOpts());
  
  // Create the target instance.
  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget()) {
    Clang->setInvocation(nullptr);
    return;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang->getTarget().adjust(Clang->getLangOpts());
  
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
  for (const auto &RemappedFile : RemappedFiles) {
    PreprocessorOpts.addRemappedFile(RemappedFile.first, RemappedFile.second);
    OwnedBuffers.push_back(RemappedFile.second);
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
  std::unique_ptr<llvm::MemoryBuffer> OverrideMainBuffer;
  if (!getPreambleFile(this).empty()) {
    std::string CompleteFilePath(File);
    llvm::sys::fs::UniqueID CompleteFileID;

    if (!llvm::sys::fs::getUniqueID(CompleteFilePath, CompleteFileID)) {
      std::string MainPath(OriginalSourceFile);
      llvm::sys::fs::UniqueID MainID;
      if (!llvm::sys::fs::getUniqueID(MainPath, MainID)) {
        if (CompleteFileID == MainID && Line > 1)
          OverrideMainBuffer = getMainBufferWithPrecompiledPreamble(
              PCHContainerOps, *CCInvocation, false, Line - 1);
      }
    }
  }

  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile,
                                     OverrideMainBuffer.get());
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PreprocessorOpts.ImplicitPCHInclude = getPreambleFile(this);
    PreprocessorOpts.DisablePCHValidation = true;

    OwnedBuffers.push_back(OverrideMainBuffer.release());
  } else {
    PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
    PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  }

  // Disable the preprocessing record if modules are not enabled.
  if (!Clang->getLangOpts().Modules)
    PreprocessorOpts.DetailedRecord = false;

  std::unique_ptr<SyntaxOnlyAction> Act;
  Act.reset(new SyntaxOnlyAction);
  if (Act->BeginSourceFile(*Clang.get(), Clang->getFrontendOpts().Inputs[0])) {
    Act->Execute();
    Act->EndSourceFile();
  }
}

bool ASTUnit::Save(StringRef File) {
  if (HadModuleLoaderFatalFailure)
    return true;

  // Write to a temporary file and later rename it to the actual file, to avoid
  // possible race conditions.
  SmallString<128> TempPath;
  TempPath = File;
  TempPath += "-%%%%%%%%";
  int fd;
  if (llvm::sys::fs::createUniqueFile(TempPath, fd, TempPath))
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

  if (llvm::sys::fs::rename(TempPath, File)) {
    llvm::sys::fs::remove(TempPath);
    return true;
  }

  return false;
}

static bool serializeUnit(ASTWriter &Writer,
                          SmallVectorImpl<char> &Buffer,
                          Sema &S,
                          bool hasErrors,
                          raw_ostream &OS) {
  Writer.WriteAST(S, std::string(), nullptr, "", hasErrors);

  // Write the generated bitstream to "Out".
  if (!Buffer.empty())
    OS.write(Buffer.data(), Buffer.size());

  return false;
}

bool ASTUnit::serialize(raw_ostream &OS) {
  // For serialization we are lenient if the errors were only warn-as-error kind.
  bool hasErrors = getDiagnostics().hasUncompilableErrorOccurred();

  if (WriterData)
    return serializeUnit(WriterData->Writer, WriterData->Buffer,
                         getSema(), hasErrors, OS);

  SmallString<128> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ASTWriter Writer(Stream, { });
  return serializeUnit(Writer, Buffer, getSema(), hasErrors, OS);
}

typedef ContinuousRangeMap<unsigned, int, 2> SLocRemap;

void ASTUnit::TranslateStoredDiagnostics(
                          FileManager &FileMgr,
                          SourceManager &SrcMgr,
                          const SmallVectorImpl<StandaloneDiagnostic> &Diags,
                          SmallVectorImpl<StoredDiagnostic> &Out) {
  // Map the standalone diagnostic into the new source manager. We also need to
  // remap all the locations to the new view. This includes the diag location,
  // any associated source ranges, and the source ranges of associated fix-its.
  // FIXME: There should be a cleaner way to do this.

  SmallVector<StoredDiagnostic, 4> Result;
  Result.reserve(Diags.size());
  for (const StandaloneDiagnostic &SD : Diags) {
    // Rebuild the StoredDiagnostic.
    if (SD.Filename.empty())
      continue;
    const FileEntry *FE = FileMgr.getFile(SD.Filename);
    if (!FE)
      continue;
    FileID FID = SrcMgr.translateFile(FE);
    SourceLocation FileLoc = SrcMgr.getLocForStartOfFile(FID);
    if (FileLoc.isInvalid())
      continue;
    SourceLocation L = FileLoc.getLocWithOffset(SD.LocOffset);
    FullSourceLoc Loc(L, SrcMgr);

    SmallVector<CharSourceRange, 4> Ranges;
    Ranges.reserve(SD.Ranges.size());
    for (const auto &Range : SD.Ranges) {
      SourceLocation BL = FileLoc.getLocWithOffset(Range.first);
      SourceLocation EL = FileLoc.getLocWithOffset(Range.second);
      Ranges.push_back(CharSourceRange::getCharRange(BL, EL));
    }

    SmallVector<FixItHint, 2> FixIts;
    FixIts.reserve(SD.FixIts.size());
    for (const StandaloneFixIt &FixIt : SD.FixIts) {
      FixIts.push_back(FixItHint());
      FixItHint &FH = FixIts.back();
      FH.CodeToInsert = FixIt.CodeToInsert;
      SourceLocation BL = FileLoc.getLocWithOffset(FixIt.RemoveRange.first);
      SourceLocation EL = FileLoc.getLocWithOffset(FixIt.RemoveRange.second);
      FH.RemoveRange = CharSourceRange::getCharRange(BL, EL);
    }

    Result.push_back(StoredDiagnostic(SD.Level, SD.ID, 
                                      SD.Message, Loc, Ranges, FixIts));
  }
  Result.swap(Out);
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
  std::tie(FID, Offset) = SM.getDecomposedLoc(FileLoc);
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

  LocDeclsTy::iterator I = std::upper_bound(Decls->begin(), Decls->end(),
                                            LocDecl, llvm::less_first());

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

  LocDeclsTy::iterator BeginIt =
      std::lower_bound(LocDecls.begin(), LocDecls.end(),
                       std::make_pair(Offset, (Decl *)nullptr),
                       llvm::less_first());
  if (BeginIt != LocDecls.begin())
    --BeginIt;

  // If we are pointing at a top-level decl inside an objc container, we need
  // to backtrack until we find it otherwise we will fail to report that the
  // region overlaps with an objc container.
  while (BeginIt != LocDecls.begin() &&
         BeginIt->second->isTopLevelDeclInObjCContainer())
    --BeginIt;

  LocDeclsTy::iterator EndIt = std::upper_bound(
      LocDecls.begin(), LocDecls.end(),
      std::make_pair(Offset + Length, (Decl *)nullptr), llvm::less_first());
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

llvm::iterator_range<PreprocessingRecord::iterator>
ASTUnit::getLocalPreprocessingEntities() const {
  if (isMainFileAST()) {
    serialization::ModuleFile &
      Mod = Reader->getModuleManager().getPrimaryModule();
    return Reader->getModulePreprocessedEntities(Mod);
  }

  if (PreprocessingRecord *PPRec = PP->getPreprocessingRecord())
    return llvm::make_range(PPRec->local_begin(), PPRec->local_end());

  return llvm::make_range(PreprocessingRecord::iterator(),
                          PreprocessingRecord::iterator());
}

bool ASTUnit::visitLocalTopLevelDecls(void *context, DeclVisitorFn Fn) {
  if (isMainFileAST()) {
    serialization::ModuleFile &
      Mod = Reader->getModuleManager().getPrimaryModule();
    for (const Decl *D : Reader->getModuleFileLevelDecls(Mod)) {
      if (!Fn(context, D))
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

const FileEntry *ASTUnit::getPCHFile() {
  if (!Reader)
    return nullptr;

  serialization::ModuleFile *Mod = nullptr;
  Reader->getModuleManager().visit([&Mod](serialization::ModuleFile &M) {
    switch (M.Kind) {
    case serialization::MK_ImplicitModule:
    case serialization::MK_ExplicitModule:
    case serialization::MK_PrebuiltModule:
      return true; // skip dependencies.
    case serialization::MK_PCH:
      Mod = &M;
      return true; // found it.
    case serialization::MK_Preamble:
      return false; // look in dependencies.
    case serialization::MK_MainFile:
      return false; // look in dependencies.
    }

    return true;
  });
  if (Mod)
    return Mod->File;

  return nullptr;
}

bool ASTUnit::isModuleFile() {
  return isMainFileAST() && !ASTFileLangOpts.CurrentModule.empty();
}

void ASTUnit::PreambleData::countLines() const {
  NumLines = 0;
  if (empty())
    return;

  NumLines = std::count(Buffer.begin(), Buffer.end(), '\n');

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

ASTUnit::ConcurrencyState::ConcurrencyState() { Mutex = nullptr; }
ASTUnit::ConcurrencyState::~ConcurrencyState() {}
void ASTUnit::ConcurrencyState::start() {}
void ASTUnit::ConcurrencyState::finish() {}

#endif // NDEBUG
