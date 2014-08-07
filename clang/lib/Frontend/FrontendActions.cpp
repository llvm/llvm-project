//===--- FrontendActions.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendActions.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>

using namespace clang;

//===----------------------------------------------------------------------===//
// Custom Actions
//===----------------------------------------------------------------------===//

ASTConsumer *InitOnlyAction::CreateASTConsumer(CompilerInstance &CI,
                                               StringRef InFile) {
  return new ASTConsumer();
}

void InitOnlyAction::ExecuteAction() {
}

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

ASTConsumer *ASTPrintAction::CreateASTConsumer(CompilerInstance &CI,
                                               StringRef InFile) {
  if (raw_ostream *OS = CI.createDefaultOutputFile(false, InFile))
    return CreateASTPrinter(OS, CI.getFrontendOpts().ASTDumpFilter);
  return nullptr;
}

ASTConsumer *ASTDumpAction::CreateASTConsumer(CompilerInstance &CI,
                                              StringRef InFile) {
  return CreateASTDumper(CI.getFrontendOpts().ASTDumpFilter,
                         CI.getFrontendOpts().ASTDumpLookups);
}

ASTConsumer *ASTDeclListAction::CreateASTConsumer(CompilerInstance &CI,
                                                  StringRef InFile) {
  return CreateASTDeclNodeLister();
}

ASTConsumer *ASTViewAction::CreateASTConsumer(CompilerInstance &CI,
                                              StringRef InFile) {
  return CreateASTViewer();
}

ASTConsumer *DeclContextPrintAction::CreateASTConsumer(CompilerInstance &CI,
                                                       StringRef InFile) {
  return CreateDeclContextPrinter();
}

ASTConsumer *GeneratePCHAction::CreateASTConsumer(CompilerInstance &CI,
                                                  StringRef InFile) {
  std::string Sysroot;
  std::string OutputFile;
  raw_ostream *OS = nullptr;
  if (ComputeASTConsumerArguments(CI, InFile, Sysroot, OutputFile, OS))
    return nullptr;

  if (!CI.getFrontendOpts().RelocatablePCH)
    Sysroot.clear();
  return new PCHGenerator(CI.getPreprocessor(), OutputFile, nullptr, Sysroot,
                          OS);
}

bool GeneratePCHAction::ComputeASTConsumerArguments(CompilerInstance &CI,
                                                    StringRef InFile,
                                                    std::string &Sysroot,
                                                    std::string &OutputFile,
                                                    raw_ostream *&OS) {
  Sysroot = CI.getHeaderSearchOpts().Sysroot;
  if (CI.getFrontendOpts().RelocatablePCH && Sysroot.empty()) {
    CI.getDiagnostics().Report(diag::err_relocatable_without_isysroot);
    return true;
  }

  // We use createOutputFile here because this is exposed via libclang, and we
  // must disable the RemoveFileOnSignal behavior.
  // We use a temporary to avoid race conditions.
  OS = CI.createOutputFile(CI.getFrontendOpts().OutputFile, /*Binary=*/true,
                           /*RemoveFileOnSignal=*/false, InFile,
                           /*Extension=*/"", /*useTemporary=*/true);
  if (!OS)
    return true;

  OutputFile = CI.getFrontendOpts().OutputFile;
  return false;
}

ASTConsumer *GenerateModuleAction::CreateASTConsumer(CompilerInstance &CI,
                                                     StringRef InFile) {
  std::string Sysroot;
  std::string OutputFile;
  raw_ostream *OS = nullptr;
  if (ComputeASTConsumerArguments(CI, InFile, Sysroot, OutputFile, OS))
    return nullptr;

  return new PCHGenerator(CI.getPreprocessor(), OutputFile, Module, 
                          Sysroot, OS);
}

static SmallVectorImpl<char> &
operator+=(SmallVectorImpl<char> &Includes, StringRef RHS) {
  Includes.append(RHS.begin(), RHS.end());
  return Includes;
}

static std::error_code addHeaderInclude(StringRef HeaderName,
                                        SmallVectorImpl<char> &Includes,
                                        const LangOptions &LangOpts,
                                        bool IsExternC) {
  if (IsExternC && LangOpts.CPlusPlus)
    Includes += "extern \"C\" {\n";
  if (LangOpts.ObjC1)
    Includes += "#import \"";
  else
    Includes += "#include \"";
  // Use an absolute path for the include; there's no reason to think that
  // a relative path will work (. might not be on our include path) or that
  // it will find the same file.
  if (llvm::sys::path::is_absolute(HeaderName)) {
    Includes += HeaderName;
  } else {
    SmallString<256> Header = HeaderName;
    if (std::error_code Err = llvm::sys::fs::make_absolute(Header))
      return Err;
    Includes += Header;
  }
  Includes += "\"\n";
  if (IsExternC && LangOpts.CPlusPlus)
    Includes += "}\n";
  return std::error_code();
}

static std::error_code addHeaderInclude(const FileEntry *Header,
                                        SmallVectorImpl<char> &Includes,
                                        const LangOptions &LangOpts,
                                        bool IsExternC) {
  return addHeaderInclude(Header->getName(), Includes, LangOpts, IsExternC);
}

/// \brief Collect the set of header includes needed to construct the given 
/// module and update the TopHeaders file set of the module.
///
/// \param Module The module we're collecting includes from.
///
/// \param Includes Will be augmented with the set of \#includes or \#imports
/// needed to load all of the named headers.
static std::error_code
collectModuleHeaderIncludes(const LangOptions &LangOpts, FileManager &FileMgr,
                            ModuleMap &ModMap, clang::Module *Module,
                            SmallVectorImpl<char> &Includes) {
  // Don't collect any headers for unavailable modules.
  if (!Module->isAvailable())
    return std::error_code();

  // Add includes for each of these headers.
  for (unsigned I = 0, N = Module->NormalHeaders.size(); I != N; ++I) {
    const FileEntry *Header = Module->NormalHeaders[I];
    Module->addTopHeader(Header);
    if (std::error_code Err =
            addHeaderInclude(Header, Includes, LangOpts, Module->IsExternC))
      return Err;
  }
  // Note that Module->PrivateHeaders will not be a TopHeader.

  if (const FileEntry *UmbrellaHeader = Module->getUmbrellaHeader()) {
    Module->addTopHeader(UmbrellaHeader);
    if (Module->Parent) {
      // Include the umbrella header for submodules.
      if (std::error_code Err = addHeaderInclude(UmbrellaHeader, Includes,
                                                 LangOpts, Module->IsExternC))
        return Err;
    }
  } else if (const DirectoryEntry *UmbrellaDir = Module->getUmbrellaDir()) {
    // Add all of the headers we find in this subdirectory.
    std::error_code EC;
    SmallString<128> DirNative;
    llvm::sys::path::native(UmbrellaDir->getName(), DirNative);
    for (llvm::sys::fs::recursive_directory_iterator Dir(DirNative.str(), EC), 
                                                     DirEnd;
         Dir != DirEnd && !EC; Dir.increment(EC)) {
      // Check whether this entry has an extension typically associated with 
      // headers.
      if (!llvm::StringSwitch<bool>(llvm::sys::path::extension(Dir->path()))
          .Cases(".h", ".H", ".hh", ".hpp", true)
          .Default(false))
        continue;
      
      // If this header is marked 'unavailable' in this module, don't include 
      // it.
      if (const FileEntry *Header = FileMgr.getFile(Dir->path())) {
        if (ModMap.isHeaderUnavailableInModule(Header, Module))
          continue;
        Module->addTopHeader(Header);
      }
      
      // Include this header as part of the umbrella directory.
      if (std::error_code Err = addHeaderInclude(Dir->path(), Includes,
                                                 LangOpts, Module->IsExternC))
        return Err;
    }

    if (EC)
      return EC;
  }
  
  // Recurse into submodules.
  for (clang::Module::submodule_iterator Sub = Module->submodule_begin(),
                                      SubEnd = Module->submodule_end();
       Sub != SubEnd; ++Sub)
    if (std::error_code Err = collectModuleHeaderIncludes(
            LangOpts, FileMgr, ModMap, *Sub, Includes))
      return Err;

  return std::error_code();
}

bool GenerateModuleAction::BeginSourceFileAction(CompilerInstance &CI, 
                                                 StringRef Filename) {
  // Find the module map file.  
  const FileEntry *ModuleMap = CI.getFileManager().getFile(Filename);
  if (!ModuleMap)  {
    CI.getDiagnostics().Report(diag::err_module_map_not_found)
      << Filename;
    return false;
  }
  
  // Parse the module map file.
  HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
  if (HS.loadModuleMapFile(ModuleMap, IsSystem))
    return false;
  
  if (CI.getLangOpts().CurrentModule.empty()) {
    CI.getDiagnostics().Report(diag::err_missing_module_name);
    
    // FIXME: Eventually, we could consider asking whether there was just
    // a single module described in the module map, and use that as a 
    // default. Then it would be fairly trivial to just "compile" a module
    // map with a single module (the common case).
    return false;
  }

  // If we're being run from the command-line, the module build stack will not
  // have been filled in yet, so complete it now in order to allow us to detect
  // module cycles.
  SourceManager &SourceMgr = CI.getSourceManager();
  if (SourceMgr.getModuleBuildStack().empty())
    SourceMgr.pushModuleBuildStack(CI.getLangOpts().CurrentModule,
                                   FullSourceLoc(SourceLocation(), SourceMgr));

  // Dig out the module definition.
  Module = HS.lookupModule(CI.getLangOpts().CurrentModule, 
                           /*AllowSearch=*/false);
  if (!Module) {
    CI.getDiagnostics().Report(diag::err_missing_module)
      << CI.getLangOpts().CurrentModule << Filename;
    
    return false;
  }

  // Check whether we can build this module at all.
  clang::Module::Requirement Requirement;
  clang::Module::HeaderDirective MissingHeader;
  if (!Module->isAvailable(CI.getLangOpts(), CI.getTarget(), Requirement,
                           MissingHeader)) {
    if (MissingHeader.FileNameLoc.isValid()) {
      CI.getDiagnostics().Report(MissingHeader.FileNameLoc,
                                 diag::err_module_header_missing)
        << MissingHeader.IsUmbrella << MissingHeader.FileName;
    } else {
      CI.getDiagnostics().Report(diag::err_module_unavailable)
        << Module->getFullModuleName()
        << Requirement.second << Requirement.first;
    }

    return false;
  }

  if (!ModuleMapForUniquing)
    ModuleMapForUniquing = ModuleMap;
  Module->ModuleMap = ModuleMapForUniquing;
  assert(Module->ModuleMap && "missing module map file");

  FileManager &FileMgr = CI.getFileManager();

  // Collect the set of #includes we need to build the module.
  SmallString<256> HeaderContents;
  std::error_code Err = std::error_code();
  if (const FileEntry *UmbrellaHeader = Module->getUmbrellaHeader())
    Err = addHeaderInclude(UmbrellaHeader, HeaderContents, CI.getLangOpts(),
                           Module->IsExternC);
  if (!Err)
    Err = collectModuleHeaderIncludes(
        CI.getLangOpts(), FileMgr,
        CI.getPreprocessor().getHeaderSearchInfo().getModuleMap(), Module,
        HeaderContents);

  if (Err) {
    CI.getDiagnostics().Report(diag::err_module_cannot_create_includes)
      << Module->getFullModuleName() << Err.message();
    return false;
  }

  llvm::MemoryBuffer *InputBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(HeaderContents,
                                           Module::getModuleInputBufferName());
  // Ownership of InputBuffer will be transferred to the SourceManager.
  setCurrentInput(FrontendInputFile(InputBuffer, getCurrentFileKind(),
                                    Module->IsSystem));
  return true;
}

bool GenerateModuleAction::ComputeASTConsumerArguments(CompilerInstance &CI,
                                                       StringRef InFile,
                                                       std::string &Sysroot,
                                                       std::string &OutputFile,
                                                       raw_ostream *&OS) {
  // If no output file was provided, figure out where this module would go
  // in the module cache.
  if (CI.getFrontendOpts().OutputFile.empty()) {
    HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
    CI.getFrontendOpts().OutputFile =
        HS.getModuleFileName(CI.getLangOpts().CurrentModule,
                             ModuleMapForUniquing->getName());
  }
  
  // We use createOutputFile here because this is exposed via libclang, and we
  // must disable the RemoveFileOnSignal behavior.
  // We use a temporary to avoid race conditions.
  OS = CI.createOutputFile(CI.getFrontendOpts().OutputFile, /*Binary=*/true,
                           /*RemoveFileOnSignal=*/false, InFile,
                           /*Extension=*/"", /*useTemporary=*/true,
                           /*CreateMissingDirectories=*/true);
  if (!OS)
    return true;
  
  OutputFile = CI.getFrontendOpts().OutputFile;
  return false;
}

ASTConsumer *SyntaxOnlyAction::CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) {
  return new ASTConsumer();
}

ASTConsumer *DumpModuleInfoAction::CreateASTConsumer(CompilerInstance &CI,
                                                     StringRef InFile) {
  return new ASTConsumer();
}

ASTConsumer *VerifyPCHAction::CreateASTConsumer(CompilerInstance &CI,
                                                StringRef InFile) {
  return new ASTConsumer();
}

void VerifyPCHAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  bool Preamble = CI.getPreprocessorOpts().PrecompiledPreambleBytes.first != 0;
  const std::string &Sysroot = CI.getHeaderSearchOpts().Sysroot;
  std::unique_ptr<ASTReader> Reader(
      new ASTReader(CI.getPreprocessor(), CI.getASTContext(),
                    Sysroot.empty() ? "" : Sysroot.c_str(),
                    /*DisableValidation*/ false,
                    /*AllowPCHWithCompilerErrors*/ false,
                    /*AllowConfigurationMismatch*/ true,
                    /*ValidateSystemInputs*/ true));

  Reader->ReadAST(getCurrentFile(),
                  Preamble ? serialization::MK_Preamble
                           : serialization::MK_PCH,
                  SourceLocation(),
                  ASTReader::ARR_ConfigurationMismatch);
}

namespace {
  /// \brief AST reader listener that dumps module information for a module
  /// file.
  class DumpModuleInfoListener : public ASTReaderListener {
    llvm::raw_ostream &Out;

  public:
    DumpModuleInfoListener(llvm::raw_ostream &Out) : Out(Out) { }

#define DUMP_BOOLEAN(Value, Text)                       \
    Out.indent(4) << Text << ": " << (Value? "Yes" : "No") << "\n"

    bool ReadFullVersionInformation(StringRef FullVersion) override {
      Out.indent(2)
        << "Generated by "
        << (FullVersion == getClangFullRepositoryVersion()? "this"
                                                          : "a different")
        << " Clang: " << FullVersion << "\n";
      return ASTReaderListener::ReadFullVersionInformation(FullVersion);
    }

    void ReadModuleName(StringRef ModuleName) override {
      Out.indent(2) << "Module name: " << ModuleName << "\n";
    }
    void ReadModuleMapFile(StringRef ModuleMapPath) override {
      Out.indent(2) << "Module map file: " << ModuleMapPath << "\n";
    }

    bool ReadLanguageOptions(const LangOptions &LangOpts,
                             bool Complain) override {
      Out.indent(2) << "Language options:\n";
#define LANGOPT(Name, Bits, Default, Description) \
      DUMP_BOOLEAN(LangOpts.Name, Description);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
      Out.indent(4) << Description << ": "                   \
                    << static_cast<unsigned>(LangOpts.get##Name()) << "\n";
#define VALUE_LANGOPT(Name, Bits, Default, Description) \
      Out.indent(4) << Description << ": " << LangOpts.Name << "\n";
#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"
      return false;
    }

    bool ReadTargetOptions(const TargetOptions &TargetOpts,
                           bool Complain) override {
      Out.indent(2) << "Target options:\n";
      Out.indent(4) << "  Triple: " << TargetOpts.Triple << "\n";
      Out.indent(4) << "  CPU: " << TargetOpts.CPU << "\n";
      Out.indent(4) << "  ABI: " << TargetOpts.ABI << "\n";

      if (!TargetOpts.FeaturesAsWritten.empty()) {
        Out.indent(4) << "Target features:\n";
        for (unsigned I = 0, N = TargetOpts.FeaturesAsWritten.size();
             I != N; ++I) {
          Out.indent(6) << TargetOpts.FeaturesAsWritten[I] << "\n";
        }
      }

      return false;
    }

    virtual bool
    ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
                          bool Complain) override {
      Out.indent(2) << "Diagnostic options:\n";
#define DIAGOPT(Name, Bits, Default) DUMP_BOOLEAN(DiagOpts->Name, #Name);
#define ENUM_DIAGOPT(Name, Type, Bits, Default) \
      Out.indent(4) << #Name << ": " << DiagOpts->get##Name() << "\n";
#define VALUE_DIAGOPT(Name, Bits, Default) \
      Out.indent(4) << #Name << ": " << DiagOpts->Name << "\n";
#include "clang/Basic/DiagnosticOptions.def"

      Out.indent(4) << "Diagnostic flags:\n";
      for (const std::string &Warning : DiagOpts->Warnings)
        Out.indent(6) << "-W" << Warning << "\n";
      for (const std::string &Remark : DiagOpts->Remarks)
        Out.indent(6) << "-R" << Remark << "\n";

      return false;
    }

    bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                                 bool Complain) override {
      Out.indent(2) << "Header search options:\n";
      Out.indent(4) << "System root [-isysroot=]: '" << HSOpts.Sysroot << "'\n";
      DUMP_BOOLEAN(HSOpts.UseBuiltinIncludes,
                   "Use builtin include directories [-nobuiltininc]");
      DUMP_BOOLEAN(HSOpts.UseStandardSystemIncludes,
                   "Use standard system include directories [-nostdinc]");
      DUMP_BOOLEAN(HSOpts.UseStandardCXXIncludes,
                   "Use standard C++ include directories [-nostdinc++]");
      DUMP_BOOLEAN(HSOpts.UseLibcxx,
                   "Use libc++ (rather than libstdc++) [-stdlib=]");
      return false;
    }

    bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                                 bool Complain,
                                 std::string &SuggestedPredefines) override {
      Out.indent(2) << "Preprocessor options:\n";
      DUMP_BOOLEAN(PPOpts.UsePredefines,
                   "Uses compiler/target-specific predefines [-undef]");
      DUMP_BOOLEAN(PPOpts.DetailedRecord,
                   "Uses detailed preprocessing record (for indexing)");

      if (!PPOpts.Macros.empty()) {
        Out.indent(4) << "Predefined macros:\n";
      }

      for (std::vector<std::pair<std::string, bool/*isUndef*/> >::const_iterator
             I = PPOpts.Macros.begin(), IEnd = PPOpts.Macros.end();
           I != IEnd; ++I) {
        Out.indent(6);
        if (I->second)
          Out << "-U";
        else
          Out << "-D";
        Out << I->first << "\n";
      }
      return false;
    }
#undef DUMP_BOOLEAN
  };
}

void DumpModuleInfoAction::ExecuteAction() {
  // Set up the output file.
  std::unique_ptr<llvm::raw_fd_ostream> OutFile;
  StringRef OutputFileName = getCompilerInstance().getFrontendOpts().OutputFile;
  if (!OutputFileName.empty() && OutputFileName != "-") {
    std::string ErrorInfo;
    OutFile.reset(new llvm::raw_fd_ostream(OutputFileName.str().c_str(),
                                           ErrorInfo, llvm::sys::fs::F_Text));
  }
  llvm::raw_ostream &Out = OutFile.get()? *OutFile.get() : llvm::outs();

  Out << "Information for module file '" << getCurrentFile() << "':\n";
  DumpModuleInfoListener Listener(Out);
  ASTReader::readASTFileControlBlock(getCurrentFile(),
                                     getCompilerInstance().getFileManager(),
                                     Listener);
}

//===----------------------------------------------------------------------===//
// Preprocessor Actions
//===----------------------------------------------------------------------===//

void DumpRawTokensAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  SourceManager &SM = PP.getSourceManager();

  // Start lexing the specified input file.
  const llvm::MemoryBuffer *FromFile = SM.getBuffer(SM.getMainFileID());
  Lexer RawLex(SM.getMainFileID(), FromFile, SM, PP.getLangOpts());
  RawLex.SetKeepWhitespaceMode(true);

  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(tok::eof)) {
    PP.DumpToken(RawTok, true);
    llvm::errs() << "\n";
    RawLex.LexFromRawLexer(RawTok);
  }
}

void DumpTokensAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  // Start preprocessing the specified input file.
  Token Tok;
  PP.EnterMainSourceFile();
  do {
    PP.Lex(Tok);
    PP.DumpToken(Tok, true);
    llvm::errs() << "\n";
  } while (Tok.isNot(tok::eof));
}

void GeneratePTHAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  if (CI.getFrontendOpts().OutputFile.empty() ||
      CI.getFrontendOpts().OutputFile == "-") {
    // FIXME: Don't fail this way.
    // FIXME: Verify that we can actually seek in the given file.
    llvm::report_fatal_error("PTH requires a seekable file for output!");
  }
  llvm::raw_fd_ostream *OS =
    CI.createDefaultOutputFile(true, getCurrentFile());
  if (!OS) return;

  CacheTokens(CI.getPreprocessor(), OS);
}

void PreprocessOnlyAction::ExecuteAction() {
  Preprocessor &PP = getCompilerInstance().getPreprocessor();

  // Ignore unknown pragmas.
  PP.IgnorePragmas();

  Token Tok;
  // Start parsing the specified input file.
  PP.EnterMainSourceFile();
  do {
    PP.Lex(Tok);
  } while (Tok.isNot(tok::eof));
}

void PrintPreprocessedAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  // Output file may need to be set to 'Binary', to avoid converting Unix style
  // line feeds (<LF>) to Microsoft style line feeds (<CR><LF>).
  //
  // Look to see what type of line endings the file uses. If there's a
  // CRLF, then we won't open the file up in binary mode. If there is
  // just an LF or CR, then we will open the file up in binary mode.
  // In this fashion, the output format should match the input format, unless
  // the input format has inconsistent line endings.
  //
  // This should be a relatively fast operation since most files won't have
  // all of their source code on a single line. However, that is still a 
  // concern, so if we scan for too long, we'll just assume the file should
  // be opened in binary mode.
  bool BinaryMode = true;
  bool InvalidFile = false;
  const SourceManager& SM = CI.getSourceManager();
  const llvm::MemoryBuffer *Buffer = SM.getBuffer(SM.getMainFileID(), 
                                                     &InvalidFile);
  if (!InvalidFile) {
    const char *cur = Buffer->getBufferStart();
    const char *end = Buffer->getBufferEnd();
    const char *next = (cur != end) ? cur + 1 : end;

    // Limit ourselves to only scanning 256 characters into the source
    // file.  This is mostly a sanity check in case the file has no 
    // newlines whatsoever.
    if (end - cur > 256) end = cur + 256;
	  
    while (next < end) {
      if (*cur == 0x0D) {  // CR
        if (*next == 0x0A)  // CRLF
          BinaryMode = false;

        break;
      } else if (*cur == 0x0A)  // LF
        break;

      ++cur, ++next;
    }
  }

  raw_ostream *OS = CI.createDefaultOutputFile(BinaryMode, getCurrentFile());
  if (!OS) return;

  DoPrintPreprocessedInput(CI.getPreprocessor(), OS,
                           CI.getPreprocessorOutputOpts());
}

void PrintPreambleAction::ExecuteAction() {
  switch (getCurrentFileKind()) {
  case IK_C:
  case IK_CXX:
  case IK_ObjC:
  case IK_ObjCXX:
  case IK_OpenCL:
  case IK_CUDA:
    break;
      
  case IK_None:
  case IK_Asm:
  case IK_PreprocessedC:
  case IK_PreprocessedCXX:
  case IK_PreprocessedObjC:
  case IK_PreprocessedObjCXX:
  case IK_AST:
  case IK_LLVM_IR:
    // We can't do anything with these.
    return;
  }
  
  CompilerInstance &CI = getCompilerInstance();
  llvm::MemoryBuffer *Buffer
      = CI.getFileManager().getBufferForFile(getCurrentFile());
  if (Buffer) {
    unsigned Preamble = Lexer::ComputePreamble(Buffer, CI.getLangOpts()).first;
    llvm::outs().write(Buffer->getBufferStart(), Preamble);
    delete Buffer;
  }
}
