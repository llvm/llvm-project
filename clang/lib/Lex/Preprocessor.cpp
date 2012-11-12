//===--- Preprocess.cpp - C Language Family Preprocessor Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Preprocessor interface.
//
//===----------------------------------------------------------------------===//
//
// Options to support:
//   -H       - Print the name of each header file used.
//   -d[DNI] - Dump various things.
//   -fworking-directory - #line's with preprocessor's working dir.
//   -fpreprocessed
//   -dependency-file,-M,-MM,-MF,-MG,-MP,-MT,-MQ,-MD,-MMD
//   -W*
//   -w
//
// Messages to emit:
//   "Multiple include guards may be useful for:\n"
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "MacroArgs.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Capacity.h"
using namespace clang;

//===----------------------------------------------------------------------===//
ExternalPreprocessorSource::~ExternalPreprocessorSource() { }

PPMutationListener::~PPMutationListener() { }

Preprocessor::Preprocessor(llvm::IntrusiveRefCntPtr<PreprocessorOptions> PPOpts,
                           DiagnosticsEngine &diags, LangOptions &opts,
                           const TargetInfo *target, SourceManager &SM,
                           HeaderSearch &Headers, ModuleLoader &TheModuleLoader,
                           IdentifierInfoLookup* IILookup,
                           bool OwnsHeaders,
                           bool DelayInitialization,
                           bool IncrProcessing)
  : PPOpts(PPOpts), Diags(&diags), LangOpts(opts), Target(target),
    FileMgr(Headers.getFileMgr()),
    SourceMgr(SM), HeaderInfo(Headers), TheModuleLoader(TheModuleLoader),
    ExternalSource(0), Identifiers(opts, IILookup), 
    IncrementalProcessing(IncrProcessing), CodeComplete(0), 
    CodeCompletionFile(0), CodeCompletionOffset(0), CodeCompletionReached(0),
    SkipMainFilePreamble(0, true), CurPPLexer(0), 
    CurDirLookup(0), CurLexerKind(CLK_Lexer), Callbacks(0), Listener(0),
    MacroArgCache(0), Record(0), MIChainHead(0), MICache(0) 
{
  OwnsHeaderSearch = OwnsHeaders;
  
  ScratchBuf = new ScratchBuffer(SourceMgr);
  CounterValue = 0; // __COUNTER__ starts at 0.
  
  // Clear stats.
  NumDirectives = NumDefined = NumUndefined = NumPragma = 0;
  NumIf = NumElse = NumEndif = 0;
  NumEnteredSourceFiles = 0;
  NumMacroExpanded = NumFnMacroExpanded = NumBuiltinMacroExpanded = 0;
  NumFastMacroExpanded = NumTokenPaste = NumFastTokenPaste = 0;
  MaxIncludeStackDepth = 0;
  NumSkipped = 0;
  
  // Default to discarding comments.
  KeepComments = false;
  KeepMacroComments = false;
  SuppressIncludeNotFoundError = false;
  
  // Macro expansion is enabled.
  DisableMacroExpansion = false;
  MacroExpansionInDirectivesOverride = false;
  InMacroArgs = false;
  InMacroArgPreExpansion = false;
  NumCachedTokenLexers = 0;
  PragmasEnabled = true;

  CachedLexPos = 0;
  
  // We haven't read anything from the external source.
  ReadMacrosFromExternalSource = false;
  
  // "Poison" __VA_ARGS__, which can only appear in the expansion of a macro.
  // This gets unpoisoned where it is allowed.
  (Ident__VA_ARGS__ = getIdentifierInfo("__VA_ARGS__"))->setIsPoisoned();
  SetPoisonReason(Ident__VA_ARGS__,diag::ext_pp_bad_vaargs_use);
  
  // Initialize the pragma handlers.
  PragmaHandlers = new PragmaNamespace(StringRef());
  RegisterBuiltinPragmas();
  
  // Initialize builtin macros like __LINE__ and friends.
  RegisterBuiltinMacros();
  
  if(LangOpts.Borland) {
    Ident__exception_info        = getIdentifierInfo("_exception_info");
    Ident___exception_info       = getIdentifierInfo("__exception_info");
    Ident_GetExceptionInfo       = getIdentifierInfo("GetExceptionInformation");
    Ident__exception_code        = getIdentifierInfo("_exception_code");
    Ident___exception_code       = getIdentifierInfo("__exception_code");
    Ident_GetExceptionCode       = getIdentifierInfo("GetExceptionCode");
    Ident__abnormal_termination  = getIdentifierInfo("_abnormal_termination");
    Ident___abnormal_termination = getIdentifierInfo("__abnormal_termination");
    Ident_AbnormalTermination    = getIdentifierInfo("AbnormalTermination");
  } else {
    Ident__exception_info = Ident__exception_code = Ident__abnormal_termination = 0;
    Ident___exception_info = Ident___exception_code = Ident___abnormal_termination = 0;
    Ident_GetExceptionInfo = Ident_GetExceptionCode = Ident_AbnormalTermination = 0;
  }

  if (!DelayInitialization) {
    assert(Target && "Must provide target information for PP initialization");
    Initialize(*Target);
  }
}

Preprocessor::~Preprocessor() {
  assert(BacktrackPositions.empty() && "EnableBacktrack/Backtrack imbalance!");

  while (!IncludeMacroStack.empty()) {
    delete IncludeMacroStack.back().TheLexer;
    delete IncludeMacroStack.back().TheTokenLexer;
    IncludeMacroStack.pop_back();
  }

  // Free any macro definitions.
  for (MacroInfoChain *I = MIChainHead ; I ; I = I->Next)
    I->MI.Destroy();

  // Free any cached macro expanders.
  for (unsigned i = 0, e = NumCachedTokenLexers; i != e; ++i)
    delete TokenLexerCache[i];

  // Free any cached MacroArgs.
  for (MacroArgs *ArgList = MacroArgCache; ArgList; )
    ArgList = ArgList->deallocate();

  // Release pragma information.
  delete PragmaHandlers;

  // Delete the scratch buffer info.
  delete ScratchBuf;

  // Delete the header search info, if we own it.
  if (OwnsHeaderSearch)
    delete &HeaderInfo;

  delete Callbacks;
}

void Preprocessor::Initialize(const TargetInfo &Target) {
  assert((!this->Target || this->Target == &Target) &&
         "Invalid override of target information");
  this->Target = &Target;
  
  // Initialize information about built-ins.
  BuiltinInfo.InitializeTarget(Target);
  HeaderInfo.setTarget(Target);
}

void Preprocessor::setPTHManager(PTHManager* pm) {
  PTH.reset(pm);
  FileMgr.addStatCache(PTH->createStatCache());
}

void Preprocessor::DumpToken(const Token &Tok, bool DumpFlags) const {
  llvm::errs() << tok::getTokenName(Tok.getKind()) << " '"
               << getSpelling(Tok) << "'";

  if (!DumpFlags) return;

  llvm::errs() << "\t";
  if (Tok.isAtStartOfLine())
    llvm::errs() << " [StartOfLine]";
  if (Tok.hasLeadingSpace())
    llvm::errs() << " [LeadingSpace]";
  if (Tok.isExpandDisabled())
    llvm::errs() << " [ExpandDisabled]";
  if (Tok.needsCleaning()) {
    const char *Start = SourceMgr.getCharacterData(Tok.getLocation());
    llvm::errs() << " [UnClean='" << StringRef(Start, Tok.getLength())
                 << "']";
  }

  llvm::errs() << "\tLoc=<";
  DumpLocation(Tok.getLocation());
  llvm::errs() << ">";
}

void Preprocessor::DumpLocation(SourceLocation Loc) const {
  Loc.dump(SourceMgr);
}

void Preprocessor::DumpMacro(const MacroInfo &MI) const {
  llvm::errs() << "MACRO: ";
  for (unsigned i = 0, e = MI.getNumTokens(); i != e; ++i) {
    DumpToken(MI.getReplacementToken(i));
    llvm::errs() << "  ";
  }
  llvm::errs() << "\n";
}

void Preprocessor::PrintStats() {
  llvm::errs() << "\n*** Preprocessor Stats:\n";
  llvm::errs() << NumDirectives << " directives found:\n";
  llvm::errs() << "  " << NumDefined << " #define.\n";
  llvm::errs() << "  " << NumUndefined << " #undef.\n";
  llvm::errs() << "  #include/#include_next/#import:\n";
  llvm::errs() << "    " << NumEnteredSourceFiles << " source files entered.\n";
  llvm::errs() << "    " << MaxIncludeStackDepth << " max include stack depth\n";
  llvm::errs() << "  " << NumIf << " #if/#ifndef/#ifdef.\n";
  llvm::errs() << "  " << NumElse << " #else/#elif.\n";
  llvm::errs() << "  " << NumEndif << " #endif.\n";
  llvm::errs() << "  " << NumPragma << " #pragma.\n";
  llvm::errs() << NumSkipped << " #if/#ifndef#ifdef regions skipped\n";

  llvm::errs() << NumMacroExpanded << "/" << NumFnMacroExpanded << "/"
             << NumBuiltinMacroExpanded << " obj/fn/builtin macros expanded, "
             << NumFastMacroExpanded << " on the fast path.\n";
  llvm::errs() << (NumFastTokenPaste+NumTokenPaste)
             << " token paste (##) operations performed, "
             << NumFastTokenPaste << " on the fast path.\n";

  llvm::errs() << "\nPreprocessor Memory: " << getTotalMemory() << "B total";

  llvm::errs() << "\n  BumpPtr: " << BP.getTotalMemory();
  llvm::errs() << "\n  Macro Expanded Tokens: "
               << llvm::capacity_in_bytes(MacroExpandedTokens);
  llvm::errs() << "\n  Predefines Buffer: " << Predefines.capacity();
  llvm::errs() << "\n  Macros: " << llvm::capacity_in_bytes(Macros);
  llvm::errs() << "\n  #pragma push_macro Info: "
               << llvm::capacity_in_bytes(PragmaPushMacroInfo);
  llvm::errs() << "\n  Poison Reasons: "
               << llvm::capacity_in_bytes(PoisonReasons);
  llvm::errs() << "\n  Comment Handlers: "
               << llvm::capacity_in_bytes(CommentHandlers) << "\n";
}

Preprocessor::macro_iterator
Preprocessor::macro_begin(bool IncludeExternalMacros) const {
  if (IncludeExternalMacros && ExternalSource &&
      !ReadMacrosFromExternalSource) {
    ReadMacrosFromExternalSource = true;
    ExternalSource->ReadDefinedMacros();
  }

  return Macros.begin();
}

size_t Preprocessor::getTotalMemory() const {
  return BP.getTotalMemory()
    + llvm::capacity_in_bytes(MacroExpandedTokens)
    + Predefines.capacity() /* Predefines buffer. */
    + llvm::capacity_in_bytes(Macros)
    + llvm::capacity_in_bytes(PragmaPushMacroInfo)
    + llvm::capacity_in_bytes(PoisonReasons)
    + llvm::capacity_in_bytes(CommentHandlers);
}

Preprocessor::macro_iterator
Preprocessor::macro_end(bool IncludeExternalMacros) const {
  if (IncludeExternalMacros && ExternalSource &&
      !ReadMacrosFromExternalSource) {
    ReadMacrosFromExternalSource = true;
    ExternalSource->ReadDefinedMacros();
  }

  return Macros.end();
}

/// \brief Compares macro tokens with a specified token value sequence.
static bool MacroDefinitionEquals(const MacroInfo *MI,
                                  llvm::ArrayRef<TokenValue> Tokens) {
  return Tokens.size() == MI->getNumTokens() &&
      std::equal(Tokens.begin(), Tokens.end(), MI->tokens_begin());
}

StringRef Preprocessor::getLastMacroWithSpelling(
                                    SourceLocation Loc,
                                    ArrayRef<TokenValue> Tokens) const {
  SourceLocation BestLocation;
  StringRef BestSpelling;
  for (Preprocessor::macro_iterator I = macro_begin(), E = macro_end();
       I != E; ++I) {
    if (!I->second->isObjectLike())
      continue;
    const MacroInfo *MI = I->second->findDefinitionAtLoc(Loc, SourceMgr);
    if (!MI)
      continue;
    if (!MacroDefinitionEquals(MI, Tokens))
      continue;
    SourceLocation Location = I->second->getDefinitionLoc();
    // Choose the macro defined latest.
    if (BestLocation.isInvalid() ||
        (Location.isValid() &&
         SourceMgr.isBeforeInTranslationUnit(BestLocation, Location))) {
      BestLocation = Location;
      BestSpelling = I->first->getName();
    }
  }
  return BestSpelling;
}

void Preprocessor::recomputeCurLexerKind() {
  if (CurLexer)
    CurLexerKind = CLK_Lexer;
  else if (CurPTHLexer)
    CurLexerKind = CLK_PTHLexer;
  else if (CurTokenLexer)
    CurLexerKind = CLK_TokenLexer;
  else 
    CurLexerKind = CLK_CachingLexer;
}

bool Preprocessor::SetCodeCompletionPoint(const FileEntry *File,
                                          unsigned CompleteLine,
                                          unsigned CompleteColumn) {
  assert(File);
  assert(CompleteLine && CompleteColumn && "Starts from 1:1");
  assert(!CodeCompletionFile && "Already set");

  using llvm::MemoryBuffer;

  // Load the actual file's contents.
  bool Invalid = false;
  const MemoryBuffer *Buffer = SourceMgr.getMemoryBufferForFile(File, &Invalid);
  if (Invalid)
    return true;

  // Find the byte position of the truncation point.
  const char *Position = Buffer->getBufferStart();
  for (unsigned Line = 1; Line < CompleteLine; ++Line) {
    for (; *Position; ++Position) {
      if (*Position != '\r' && *Position != '\n')
        continue;

      // Eat \r\n or \n\r as a single line.
      if ((Position[1] == '\r' || Position[1] == '\n') &&
          Position[0] != Position[1])
        ++Position;
      ++Position;
      break;
    }
  }

  Position += CompleteColumn - 1;

  // Insert '\0' at the code-completion point.
  if (Position < Buffer->getBufferEnd()) {
    CodeCompletionFile = File;
    CodeCompletionOffset = Position - Buffer->getBufferStart();

    MemoryBuffer *NewBuffer =
        MemoryBuffer::getNewUninitMemBuffer(Buffer->getBufferSize() + 1,
                                            Buffer->getBufferIdentifier());
    char *NewBuf = const_cast<char*>(NewBuffer->getBufferStart());
    char *NewPos = std::copy(Buffer->getBufferStart(), Position, NewBuf);
    *NewPos = '\0';
    std::copy(Position, Buffer->getBufferEnd(), NewPos+1);
    SourceMgr.overrideFileContents(File, NewBuffer);
  }

  return false;
}

void Preprocessor::CodeCompleteNaturalLanguage() {
  if (CodeComplete)
    CodeComplete->CodeCompleteNaturalLanguage();
  setCodeCompletionReached();
}

/// getSpelling - This method is used to get the spelling of a token into a
/// SmallVector. Note that the returned StringRef may not point to the
/// supplied buffer if a copy can be avoided.
StringRef Preprocessor::getSpelling(const Token &Tok,
                                          SmallVectorImpl<char> &Buffer,
                                          bool *Invalid) const {
  // NOTE: this has to be checked *before* testing for an IdentifierInfo.
  if (Tok.isNot(tok::raw_identifier)) {
    // Try the fast path.
    if (const IdentifierInfo *II = Tok.getIdentifierInfo())
      return II->getName();
  }

  // Resize the buffer if we need to copy into it.
  if (Tok.needsCleaning())
    Buffer.resize(Tok.getLength());

  const char *Ptr = Buffer.data();
  unsigned Len = getSpelling(Tok, Ptr, Invalid);
  return StringRef(Ptr, Len);
}

/// CreateString - Plop the specified string into a scratch buffer and return a
/// location for it.  If specified, the source location provides a source
/// location for the token.
void Preprocessor::CreateString(StringRef Str, Token &Tok,
                                SourceLocation ExpansionLocStart,
                                SourceLocation ExpansionLocEnd) {
  Tok.setLength(Str.size());

  const char *DestPtr;
  SourceLocation Loc = ScratchBuf->getToken(Str.data(), Str.size(), DestPtr);

  if (ExpansionLocStart.isValid())
    Loc = SourceMgr.createExpansionLoc(Loc, ExpansionLocStart,
                                       ExpansionLocEnd, Str.size());
  Tok.setLocation(Loc);

  // If this is a raw identifier or a literal token, set the pointer data.
  if (Tok.is(tok::raw_identifier))
    Tok.setRawIdentifierData(DestPtr);
  else if (Tok.isLiteral())
    Tok.setLiteralData(DestPtr);
}

Module *Preprocessor::getCurrentModule() {
  if (getLangOpts().CurrentModule.empty())
    return 0;
  
  return getHeaderSearchInfo().lookupModule(getLangOpts().CurrentModule);
}

//===----------------------------------------------------------------------===//
// Preprocessor Initialization Methods
//===----------------------------------------------------------------------===//


/// EnterMainSourceFile - Enter the specified FileID as the main source file,
/// which implicitly adds the builtin defines etc.
void Preprocessor::EnterMainSourceFile() {
  // We do not allow the preprocessor to reenter the main file.  Doing so will
  // cause FileID's to accumulate information from both runs (e.g. #line
  // information) and predefined macros aren't guaranteed to be set properly.
  assert(NumEnteredSourceFiles == 0 && "Cannot reenter the main file!");
  FileID MainFileID = SourceMgr.getMainFileID();

  // If MainFileID is loaded it means we loaded an AST file, no need to enter
  // a main file.
  if (!SourceMgr.isLoadedFileID(MainFileID)) {
    // Enter the main file source buffer.
    EnterSourceFile(MainFileID, 0, SourceLocation());
  
    // If we've been asked to skip bytes in the main file (e.g., as part of a
    // precompiled preamble), do so now.
    if (SkipMainFilePreamble.first > 0)
      CurLexer->SkipBytes(SkipMainFilePreamble.first, 
                          SkipMainFilePreamble.second);
    
    // Tell the header info that the main file was entered.  If the file is later
    // #imported, it won't be re-entered.
    if (const FileEntry *FE = SourceMgr.getFileEntryForID(MainFileID))
      HeaderInfo.IncrementIncludeCount(FE);
  }

  // Preprocess Predefines to populate the initial preprocessor state.
  llvm::MemoryBuffer *SB =
    llvm::MemoryBuffer::getMemBufferCopy(Predefines, "<built-in>");
  assert(SB && "Cannot create predefined source buffer");
  FileID FID = SourceMgr.createFileIDForMemBuffer(SB);
  assert(!FID.isInvalid() && "Could not create FileID for predefines?");

  // Start parsing the predefines.
  EnterSourceFile(FID, 0, SourceLocation());
}

void Preprocessor::EndSourceFile() {
  // Notify the client that we reached the end of the source file.
  if (Callbacks)
    Callbacks->EndOfMainFile();
}

//===----------------------------------------------------------------------===//
// Lexer Event Handling.
//===----------------------------------------------------------------------===//

/// LookUpIdentifierInfo - Given a tok::raw_identifier token, look up the
/// identifier information for the token and install it into the token,
/// updating the token kind accordingly.
IdentifierInfo *Preprocessor::LookUpIdentifierInfo(Token &Identifier) const {
  assert(Identifier.getRawIdentifierData() != 0 && "No raw identifier data!");

  // Look up this token, see if it is a macro, or if it is a language keyword.
  IdentifierInfo *II;
  if (!Identifier.needsCleaning()) {
    // No cleaning needed, just use the characters from the lexed buffer.
    II = getIdentifierInfo(StringRef(Identifier.getRawIdentifierData(),
                                           Identifier.getLength()));
  } else {
    // Cleaning needed, alloca a buffer, clean into it, then use the buffer.
    SmallString<64> IdentifierBuffer;
    StringRef CleanedStr = getSpelling(Identifier, IdentifierBuffer);
    II = getIdentifierInfo(CleanedStr);
  }

  // Update the token info (identifier info and appropriate token kind).
  Identifier.setIdentifierInfo(II);
  Identifier.setKind(II->getTokenID());

  return II;
}

void Preprocessor::SetPoisonReason(IdentifierInfo *II, unsigned DiagID) {
  PoisonReasons[II] = DiagID;
}

void Preprocessor::PoisonSEHIdentifiers(bool Poison) {
  assert(Ident__exception_code && Ident__exception_info);
  assert(Ident___exception_code && Ident___exception_info);
  Ident__exception_code->setIsPoisoned(Poison);
  Ident___exception_code->setIsPoisoned(Poison);
  Ident_GetExceptionCode->setIsPoisoned(Poison);
  Ident__exception_info->setIsPoisoned(Poison);
  Ident___exception_info->setIsPoisoned(Poison);
  Ident_GetExceptionInfo->setIsPoisoned(Poison);
  Ident__abnormal_termination->setIsPoisoned(Poison);
  Ident___abnormal_termination->setIsPoisoned(Poison);
  Ident_AbnormalTermination->setIsPoisoned(Poison);
}

void Preprocessor::HandlePoisonedIdentifier(Token & Identifier) {
  assert(Identifier.getIdentifierInfo() &&
         "Can't handle identifiers without identifier info!");
  llvm::DenseMap<IdentifierInfo*,unsigned>::const_iterator it =
    PoisonReasons.find(Identifier.getIdentifierInfo());
  if(it == PoisonReasons.end())
    Diag(Identifier, diag::err_pp_used_poisoned_id);
  else
    Diag(Identifier,it->second) << Identifier.getIdentifierInfo();
}

/// HandleIdentifier - This callback is invoked when the lexer reads an
/// identifier.  This callback looks up the identifier in the map and/or
/// potentially macro expands it or turns it into a named token (like 'for').
///
/// Note that callers of this method are guarded by checking the
/// IdentifierInfo's 'isHandleIdentifierCase' bit.  If this method changes, the
/// IdentifierInfo methods that compute these properties will need to change to
/// match.
void Preprocessor::HandleIdentifier(Token &Identifier) {
  assert(Identifier.getIdentifierInfo() &&
         "Can't handle identifiers without identifier info!");

  IdentifierInfo &II = *Identifier.getIdentifierInfo();

  // If the information about this identifier is out of date, update it from
  // the external source.
  // We have to treat __VA_ARGS__ in a special way, since it gets
  // serialized with isPoisoned = true, but our preprocessor may have
  // unpoisoned it if we're defining a C99 macro.
  if (II.isOutOfDate()) {
    bool CurrentIsPoisoned = false;
    if (&II == Ident__VA_ARGS__)
      CurrentIsPoisoned = Ident__VA_ARGS__->isPoisoned();

    ExternalSource->updateOutOfDateIdentifier(II);
    Identifier.setKind(II.getTokenID());

    if (&II == Ident__VA_ARGS__)
      II.setIsPoisoned(CurrentIsPoisoned);
  }
  
  // If this identifier was poisoned, and if it was not produced from a macro
  // expansion, emit an error.
  if (II.isPoisoned() && CurPPLexer) {
    HandlePoisonedIdentifier(Identifier);
  }

  // If this is a macro to be expanded, do it.
  if (MacroInfo *MI = getMacroInfo(&II)) {
    if (!DisableMacroExpansion) {
      if (Identifier.isExpandDisabled()) {
        Diag(Identifier, diag::pp_disabled_macro_expansion);
      } else if (MI->isEnabled()) {
        if (!HandleMacroExpandedIdentifier(Identifier, MI))
          return;
      } else {
        // C99 6.10.3.4p2 says that a disabled macro may never again be
        // expanded, even if it's in a context where it could be expanded in the
        // future.
        Identifier.setFlag(Token::DisableExpand);
        Diag(Identifier, diag::pp_disabled_macro_expansion);
      }
    }
  }

  // If this identifier is a keyword in C++11, produce a warning. Don't warn if
  // we're not considering macro expansion, since this identifier might be the
  // name of a macro.
  // FIXME: This warning is disabled in cases where it shouldn't be, like
  //   "#define constexpr constexpr", "int constexpr;"
  if (II.isCXX11CompatKeyword() & !DisableMacroExpansion) {
    Diag(Identifier, diag::warn_cxx11_keyword) << II.getName();
    // Don't diagnose this keyword again in this translation unit.
    II.setIsCXX11CompatKeyword(false);
  }

  // C++ 2.11p2: If this is an alternative representation of a C++ operator,
  // then we act as if it is the actual operator and not the textual
  // representation of it.
  if (II.isCPlusPlusOperatorKeyword())
    Identifier.setIdentifierInfo(0);

  // If this is an extension token, diagnose its use.
  // We avoid diagnosing tokens that originate from macro definitions.
  // FIXME: This warning is disabled in cases where it shouldn't be,
  // like "#define TY typeof", "TY(1) x".
  if (II.isExtensionToken() && !DisableMacroExpansion)
    Diag(Identifier, diag::ext_token_used);
  
  // If this is the '__experimental_modules_import' contextual keyword, note
  // that the next token indicates a module name.
  //
  // Note that we do not treat '__experimental_modules_import' as a contextual
  // keyword when we're in a caching lexer, because caching lexers only get
  // used in contexts where import declarations are disallowed.
  if (II.isModulesImport() && !InMacroArgs && !DisableMacroExpansion &&
      getLangOpts().Modules && CurLexerKind != CLK_CachingLexer) {
    ModuleImportLoc = Identifier.getLocation();
    ModuleImportPath.clear();
    ModuleImportExpectsIdentifier = true;
    CurLexerKind = CLK_LexAfterModuleImport;
  }
}

/// \brief Lex a token following the 'import' contextual keyword.
///
void Preprocessor::LexAfterModuleImport(Token &Result) {
  // Figure out what kind of lexer we actually have.
  recomputeCurLexerKind();
  
  // Lex the next token.
  Lex(Result);

  // The token sequence 
  //
  //   import identifier (. identifier)*
  //
  // indicates a module import directive. We already saw the 'import' 
  // contextual keyword, so now we're looking for the identifiers.
  if (ModuleImportExpectsIdentifier && Result.getKind() == tok::identifier) {
    // We expected to see an identifier here, and we did; continue handling
    // identifiers.
    ModuleImportPath.push_back(std::make_pair(Result.getIdentifierInfo(),
                                              Result.getLocation()));
    ModuleImportExpectsIdentifier = false;
    CurLexerKind = CLK_LexAfterModuleImport;
    return;
  }
  
  // If we're expecting a '.' or a ';', and we got a '.', then wait until we
  // see the next identifier.
  if (!ModuleImportExpectsIdentifier && Result.getKind() == tok::period) {
    ModuleImportExpectsIdentifier = true;
    CurLexerKind = CLK_LexAfterModuleImport;
    return;
  }

  // If we have a non-empty module path, load the named module.
  if (!ModuleImportPath.empty()) {
    Module *Imported = TheModuleLoader.loadModule(ModuleImportLoc,
                                                  ModuleImportPath,
                                                  Module::MacrosVisible,
                                                  /*IsIncludeDirective=*/false);
    if (Callbacks)
      Callbacks->moduleImport(ModuleImportLoc, ModuleImportPath, Imported);
  }
}

void Preprocessor::addCommentHandler(CommentHandler *Handler) {
  assert(Handler && "NULL comment handler");
  assert(std::find(CommentHandlers.begin(), CommentHandlers.end(), Handler) ==
         CommentHandlers.end() && "Comment handler already registered");
  CommentHandlers.push_back(Handler);
}

void Preprocessor::removeCommentHandler(CommentHandler *Handler) {
  std::vector<CommentHandler *>::iterator Pos
  = std::find(CommentHandlers.begin(), CommentHandlers.end(), Handler);
  assert(Pos != CommentHandlers.end() && "Comment handler not registered");
  CommentHandlers.erase(Pos);
}

bool Preprocessor::HandleComment(Token &result, SourceRange Comment) {
  bool AnyPendingTokens = false;
  for (std::vector<CommentHandler *>::iterator H = CommentHandlers.begin(),
       HEnd = CommentHandlers.end();
       H != HEnd; ++H) {
    if ((*H)->HandleComment(*this, Comment))
      AnyPendingTokens = true;
  }
  if (!AnyPendingTokens || getCommentRetentionState())
    return false;
  Lex(result);
  return true;
}

ModuleLoader::~ModuleLoader() { }

CommentHandler::~CommentHandler() { }

CodeCompletionHandler::~CodeCompletionHandler() { }

void Preprocessor::createPreprocessingRecord(bool RecordConditionalDirectives) {
  if (Record)
    return;
  
  Record = new PreprocessingRecord(getSourceManager(),
                                   RecordConditionalDirectives);
  addPPCallbacks(Record);
}
