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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

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
  return 0;
}

ASTConsumer *ASTDumpAction::CreateASTConsumer(CompilerInstance &CI,
                                              StringRef InFile) {
  return CreateASTDumper(CI.getFrontendOpts().ASTDumpFilter);
}

ASTConsumer *ASTDeclListAction::CreateASTConsumer(CompilerInstance &CI,
                                                  StringRef InFile) {
  return CreateASTDeclNodeLister();
}

ASTConsumer *ASTDumpXMLAction::CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) {
  raw_ostream *OS;
  if (CI.getFrontendOpts().OutputFile.empty())
    OS = &llvm::outs();
  else
    OS = CI.createDefaultOutputFile(false, InFile);
  if (!OS) return 0;
  return CreateASTDumperXML(*OS);
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
  raw_ostream *OS = 0;
  if (ComputeASTConsumerArguments(CI, InFile, Sysroot, OutputFile, OS))
    return 0;

  if (!CI.getFrontendOpts().RelocatablePCH)
    Sysroot.clear();
  return new PCHGenerator(CI.getPreprocessor(), OutputFile, 0, Sysroot, OS);
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
  raw_ostream *OS = 0;
  if (ComputeASTConsumerArguments(CI, InFile, Sysroot, OutputFile, OS))
    return 0;
  
  return new PCHGenerator(CI.getPreprocessor(), OutputFile, Module, 
                          Sysroot, OS);
}

static SmallVectorImpl<char> &
operator+=(SmallVectorImpl<char> &Includes, StringRef RHS) {
  Includes.append(RHS.begin(), RHS.end());
  return Includes;
}

static void addHeaderInclude(StringRef HeaderName,
                             SmallVectorImpl<char> &Includes,
                             const LangOptions &LangOpts) {
  if (LangOpts.ObjC1)
    Includes += "#import \"";
  else
    Includes += "#include \"";
  Includes += HeaderName;
  Includes += "\"\n";
}

static void addHeaderInclude(const FileEntry *Header,
                             SmallVectorImpl<char> &Includes,
                             const LangOptions &LangOpts) {
  addHeaderInclude(Header->getName(), Includes, LangOpts);
}

/// \brief Collect the set of header includes needed to construct the given 
/// module and update the TopHeaders file set of the module.
///
/// \param Module The module we're collecting includes from.
///
/// \param Includes Will be augmented with the set of \#includes or \#imports
/// needed to load all of the named headers.
static void collectModuleHeaderIncludes(const LangOptions &LangOpts,
                                        FileManager &FileMgr,
                                        ModuleMap &ModMap,
                                        clang::Module *Module,
                                        SmallVectorImpl<char> &Includes) {
  // Don't collect any headers for unavailable modules.
  if (!Module->isAvailable())
    return;

  // Add includes for each of these headers.
  for (unsigned I = 0, N = Module->Headers.size(); I != N; ++I) {
    const FileEntry *Header = Module->Headers[I];
    Module->TopHeaders.insert(Header);
    addHeaderInclude(Header, Includes, LangOpts);
  }

  if (const FileEntry *UmbrellaHeader = Module->getUmbrellaHeader()) {
    Module->TopHeaders.insert(UmbrellaHeader);
    if (Module->Parent) {
      // Include the umbrella header for submodules.
      addHeaderInclude(UmbrellaHeader, Includes, LangOpts);
    }
  } else if (const DirectoryEntry *UmbrellaDir = Module->getUmbrellaDir()) {
    // Add all of the headers we find in this subdirectory.
    llvm::error_code EC;
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
        if (ModMap.isHeaderInUnavailableModule(Header))
          continue;
        Module->TopHeaders.insert(Header);
      }
      
      // Include this header umbrella header for submodules.
      addHeaderInclude(Dir->path(), Includes, LangOpts);
    }
  }
  
  // Recurse into submodules.
  for (clang::Module::submodule_iterator Sub = Module->submodule_begin(),
                                      SubEnd = Module->submodule_end();
       Sub != SubEnd; ++Sub)
    collectModuleHeaderIncludes(LangOpts, FileMgr, ModMap, *Sub, Includes);
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
  if (HS.loadModuleMapFile(ModuleMap))
    return false;
  
  if (CI.getLangOpts().CurrentModule.empty()) {
    CI.getDiagnostics().Report(diag::err_missing_module_name);
    
    // FIXME: Eventually, we could consider asking whether there was just
    // a single module described in the module map, and use that as a 
    // default. Then it would be fairly trivial to just "compile" a module
    // map with a single module (the common case).
    return false;
  }
  
  // Dig out the module definition.
  Module = HS.lookupModule(CI.getLangOpts().CurrentModule, 
                           /*AllowSearch=*/false);
  if (!Module) {
    CI.getDiagnostics().Report(diag::err_missing_module)
      << CI.getLangOpts().CurrentModule << Filename;
    
    return false;
  }

  // Check whether we can build this module at all.
  StringRef Feature;
  if (!Module->isAvailable(CI.getLangOpts(), CI.getTarget(), Feature)) {
    CI.getDiagnostics().Report(diag::err_module_unavailable)
      << Module->getFullModuleName()
      << Feature;

    return false;
  }

  FileManager &FileMgr = CI.getFileManager();

  // Collect the set of #includes we need to build the module.
  SmallString<256> HeaderContents;
  if (const FileEntry *UmbrellaHeader = Module->getUmbrellaHeader())
    addHeaderInclude(UmbrellaHeader, HeaderContents, CI.getLangOpts());
  collectModuleHeaderIncludes(CI.getLangOpts(), FileMgr,
    CI.getPreprocessor().getHeaderSearchInfo().getModuleMap(),
    Module, HeaderContents);

  StringRef InputName = Module::getModuleInputBufferName();

  // We consistently construct a buffer as input to build the module.
  // This means the main file for modules will always be a virtual one.
  // FIXME: Maybe allow using a memory buffer as input directly instead of
  // messing with virtual files.
  const FileEntry *HeaderFile = FileMgr.getVirtualFile(InputName, 
                                                       HeaderContents.size(), 
                                                       time(0));
  llvm::MemoryBuffer *HeaderContentsBuf
    = llvm::MemoryBuffer::getMemBufferCopy(HeaderContents);
  CI.getSourceManager().overrideFileContents(HeaderFile, HeaderContentsBuf);  
  setCurrentInput(FrontendInputFile(InputName, getCurrentFileKind(),
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
    SmallString<256> ModuleFileName(HS.getModuleCachePath());
    llvm::sys::path::append(ModuleFileName, 
                            CI.getLangOpts().CurrentModule + ".pcm");
    CI.getFrontendOpts().OutputFile = ModuleFileName.str();
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
  PP.AddPragmaHandler(new EmptyPragmaHandler());

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
