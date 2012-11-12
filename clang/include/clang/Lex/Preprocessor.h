//===--- Preprocessor.h - C Language Family Preprocessor --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Preprocessor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PREPROCESSOR_H
#define LLVM_CLANG_LEX_PREPROCESSOR_H

#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PTHLexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/PPMutationListener.h"
#include "clang/Lex/TokenLexer.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace llvm {
  template<unsigned InternalLen> class SmallString;
}

namespace clang {

class SourceManager;
class ExternalPreprocessorSource;
class FileManager;
class FileEntry;
class HeaderSearch;
class PragmaNamespace;
class PragmaHandler;
class CommentHandler;
class ScratchBuffer;
class TargetInfo;
class PPCallbacks;
class CodeCompletionHandler;
class DirectoryLookup;
class PreprocessingRecord;
class ModuleLoader;
class PreprocessorOptions;

/// \brief Stores token information for comparing actual tokens with
/// predefined values.  Only handles simple tokens and identifiers.
class TokenValue {
  tok::TokenKind Kind;
  IdentifierInfo *II;

public:
  TokenValue(tok::TokenKind Kind) : Kind(Kind), II(0) {
    assert(Kind != tok::raw_identifier && "Raw identifiers are not supported.");
    assert(Kind != tok::identifier &&
           "Identifiers should be created by TokenValue(IdentifierInfo *)");
    assert(!tok::isLiteral(Kind) && "Literals are not supported.");
    assert(!tok::isAnnotation(Kind) && "Annotations are not supported.");
  }
  TokenValue(IdentifierInfo *II) : Kind(tok::identifier), II(II) {}
  bool operator==(const Token &Tok) const {
    return Tok.getKind() == Kind &&
        (!II || II == Tok.getIdentifierInfo());
  }
};

/// Preprocessor - This object engages in a tight little dance with the lexer to
/// efficiently preprocess tokens.  Lexers know only about tokens within a
/// single source file, and don't know anything about preprocessor-level issues
/// like the \#include stack, token expansion, etc.
///
class Preprocessor : public RefCountedBase<Preprocessor> {
  llvm::IntrusiveRefCntPtr<PreprocessorOptions> PPOpts;
  DiagnosticsEngine        *Diags;
  LangOptions       &LangOpts;
  const TargetInfo  *Target;
  FileManager       &FileMgr;
  SourceManager     &SourceMgr;
  ScratchBuffer     *ScratchBuf;
  HeaderSearch      &HeaderInfo;
  ModuleLoader      &TheModuleLoader;

  /// \brief External source of macros.
  ExternalPreprocessorSource *ExternalSource;


  /// PTH - An optional PTHManager object used for getting tokens from
  ///  a token cache rather than lexing the original source file.
  OwningPtr<PTHManager> PTH;

  /// BP - A BumpPtrAllocator object used to quickly allocate and release
  ///  objects internal to the Preprocessor.
  llvm::BumpPtrAllocator BP;

  /// Identifiers for builtin macros and other builtins.
  IdentifierInfo *Ident__LINE__, *Ident__FILE__;   // __LINE__, __FILE__
  IdentifierInfo *Ident__DATE__, *Ident__TIME__;   // __DATE__, __TIME__
  IdentifierInfo *Ident__INCLUDE_LEVEL__;          // __INCLUDE_LEVEL__
  IdentifierInfo *Ident__BASE_FILE__;              // __BASE_FILE__
  IdentifierInfo *Ident__TIMESTAMP__;              // __TIMESTAMP__
  IdentifierInfo *Ident__COUNTER__;                // __COUNTER__
  IdentifierInfo *Ident_Pragma, *Ident__pragma;    // _Pragma, __pragma
  IdentifierInfo *Ident__VA_ARGS__;                // __VA_ARGS__
  IdentifierInfo *Ident__has_feature;              // __has_feature
  IdentifierInfo *Ident__has_extension;            // __has_extension
  IdentifierInfo *Ident__has_builtin;              // __has_builtin
  IdentifierInfo *Ident__has_attribute;            // __has_attribute
  IdentifierInfo *Ident__has_include;              // __has_include
  IdentifierInfo *Ident__has_include_next;         // __has_include_next
  IdentifierInfo *Ident__has_warning;              // __has_warning
  IdentifierInfo *Ident__building_module;          // __building_module
  IdentifierInfo *Ident__MODULE__;                 // __MODULE__

  SourceLocation DATELoc, TIMELoc;
  unsigned CounterValue;  // Next __COUNTER__ value.

  enum {
    /// MaxIncludeStackDepth - Maximum depth of \#includes.
    MaxAllowedIncludeStackDepth = 200
  };

  // State that is set before the preprocessor begins.
  bool KeepComments : 1;
  bool KeepMacroComments : 1;
  bool SuppressIncludeNotFoundError : 1;

  // State that changes while the preprocessor runs:
  bool InMacroArgs : 1;            // True if parsing fn macro invocation args.

  /// Whether the preprocessor owns the header search object.
  bool OwnsHeaderSearch : 1;

  /// DisableMacroExpansion - True if macro expansion is disabled.
  bool DisableMacroExpansion : 1;

  /// MacroExpansionInDirectivesOverride - Temporarily disables
  /// DisableMacroExpansion (i.e. enables expansion) when parsing preprocessor
  /// directives.
  bool MacroExpansionInDirectivesOverride : 1;

  class ResetMacroExpansionHelper;

  /// \brief Whether we have already loaded macros from the external source.
  mutable bool ReadMacrosFromExternalSource : 1;

  /// \brief True if pragmas are enabled.
  bool PragmasEnabled : 1;

  /// \brief True if we are pre-expanding macro arguments.
  bool InMacroArgPreExpansion;

  /// Identifiers - This is mapping/lookup information for all identifiers in
  /// the program, including program keywords.
  mutable IdentifierTable Identifiers;

  /// Selectors - This table contains all the selectors in the program. Unlike
  /// IdentifierTable above, this table *isn't* populated by the preprocessor.
  /// It is declared/expanded here because it's role/lifetime is
  /// conceptually similar the IdentifierTable. In addition, the current control
  /// flow (in clang::ParseAST()), make it convenient to put here.
  /// FIXME: Make sure the lifetime of Identifiers/Selectors *isn't* tied to
  /// the lifetime of the preprocessor.
  SelectorTable Selectors;

  /// BuiltinInfo - Information about builtins.
  Builtin::Context BuiltinInfo;

  /// PragmaHandlers - This tracks all of the pragmas that the client registered
  /// with this preprocessor.
  PragmaNamespace *PragmaHandlers;

  /// \brief Tracks all of the comment handlers that the client registered
  /// with this preprocessor.
  std::vector<CommentHandler *> CommentHandlers;

  /// \brief True if we want to ignore EOF token and continue later on (thus 
  /// avoid tearing the Lexer and etc. down).
  bool IncrementalProcessing;

  /// \brief The code-completion handler.
  CodeCompletionHandler *CodeComplete;

  /// \brief The file that we're performing code-completion for, if any.
  const FileEntry *CodeCompletionFile;

  /// \brief The offset in file for the code-completion point.
  unsigned CodeCompletionOffset;

  /// \brief The location for the code-completion point. This gets instantiated
  /// when the CodeCompletionFile gets \#include'ed for preprocessing.
  SourceLocation CodeCompletionLoc;

  /// \brief The start location for the file of the code-completion point.
  ///
  /// This gets instantiated when the CodeCompletionFile gets \#include'ed
  /// for preprocessing.
  SourceLocation CodeCompletionFileLoc;

  /// \brief The source location of the 'import' contextual keyword we just 
  /// lexed, if any.
  SourceLocation ModuleImportLoc;

  /// \brief The module import path that we're currently processing.
  llvm::SmallVector<std::pair<IdentifierInfo *, SourceLocation>, 2> 
    ModuleImportPath;
  
  /// \brief Whether the module import expectes an identifier next. Otherwise,
  /// it expects a '.' or ';'.
  bool ModuleImportExpectsIdentifier;
  
  /// \brief The source location of the currently-active
  /// #pragma clang arc_cf_code_audited begin.
  SourceLocation PragmaARCCFCodeAuditedLoc;

  /// \brief True if we hit the code-completion point.
  bool CodeCompletionReached;

  /// \brief The number of bytes that we will initially skip when entering the
  /// main file, which is used when loading a precompiled preamble, along
  /// with a flag that indicates whether skipping this number of bytes will
  /// place the lexer at the start of a line.
  std::pair<unsigned, bool> SkipMainFilePreamble;

  /// CurLexer - This is the current top of the stack that we're lexing from if
  /// not expanding a macro and we are lexing directly from source code.
  ///  Only one of CurLexer, CurPTHLexer, or CurTokenLexer will be non-null.
  OwningPtr<Lexer> CurLexer;

  /// CurPTHLexer - This is the current top of stack that we're lexing from if
  ///  not expanding from a macro and we are lexing from a PTH cache.
  ///  Only one of CurLexer, CurPTHLexer, or CurTokenLexer will be non-null.
  OwningPtr<PTHLexer> CurPTHLexer;

  /// CurPPLexer - This is the current top of the stack what we're lexing from
  ///  if not expanding a macro.  This is an alias for either CurLexer or
  ///  CurPTHLexer.
  PreprocessorLexer *CurPPLexer;

  /// CurLookup - The DirectoryLookup structure used to find the current
  /// FileEntry, if CurLexer is non-null and if applicable.  This allows us to
  /// implement \#include_next and find directory-specific properties.
  const DirectoryLookup *CurDirLookup;

  /// CurTokenLexer - This is the current macro we are expanding, if we are
  /// expanding a macro.  One of CurLexer and CurTokenLexer must be null.
  OwningPtr<TokenLexer> CurTokenLexer;

  /// \brief The kind of lexer we're currently working with.
  enum CurLexerKind {
    CLK_Lexer,
    CLK_PTHLexer,
    CLK_TokenLexer,
    CLK_CachingLexer,
    CLK_LexAfterModuleImport
  } CurLexerKind;

  /// IncludeMacroStack - This keeps track of the stack of files currently
  /// \#included, and macros currently being expanded from, not counting
  /// CurLexer/CurTokenLexer.
  struct IncludeStackInfo {
    enum CurLexerKind     CurLexerKind;
    Lexer                 *TheLexer;
    PTHLexer              *ThePTHLexer;
    PreprocessorLexer     *ThePPLexer;
    TokenLexer            *TheTokenLexer;
    const DirectoryLookup *TheDirLookup;

    IncludeStackInfo(enum CurLexerKind K, Lexer *L, PTHLexer* P,
                     PreprocessorLexer* PPL,
                     TokenLexer* TL, const DirectoryLookup *D)
      : CurLexerKind(K), TheLexer(L), ThePTHLexer(P), ThePPLexer(PPL),
        TheTokenLexer(TL), TheDirLookup(D) {}
  };
  std::vector<IncludeStackInfo> IncludeMacroStack;

  /// Callbacks - These are actions invoked when some preprocessor activity is
  /// encountered (e.g. a file is \#included, etc).
  PPCallbacks *Callbacks;

  /// \brief Listener whose actions are invoked when an entity in the
  /// preprocessor (e.g., a macro) that was loaded from an AST file is
  /// later mutated.
  PPMutationListener *Listener;

  struct MacroExpandsInfo {
    Token Tok;
    MacroInfo *MI;
    SourceRange Range;
    MacroExpandsInfo(Token Tok, MacroInfo *MI, SourceRange Range)
      : Tok(Tok), MI(MI), Range(Range) { }
  };
  SmallVector<MacroExpandsInfo, 2> DelayedMacroExpandsCallbacks;

  /// Macros - For each IdentifierInfo that was associated with a macro, we
  /// keep a mapping to the history of all macro definitions and #undefs in
  /// the reverse order (the latest one is in the head of the list).
  llvm::DenseMap<IdentifierInfo*, MacroInfo*> Macros;
  friend class ASTReader;
  
  /// \brief Macros that we want to warn because they are not used at the end
  /// of the translation unit; we store just their SourceLocations instead
  /// something like MacroInfo*. The benefit of this is that when we are
  /// deserializing from PCH, we don't need to deserialize identifier & macros
  /// just so that we can report that they are unused, we just warn using
  /// the SourceLocations of this set (that will be filled by the ASTReader).
  /// We are using SmallPtrSet instead of a vector for faster removal.
  typedef llvm::SmallPtrSet<SourceLocation, 32> WarnUnusedMacroLocsTy;
  WarnUnusedMacroLocsTy WarnUnusedMacroLocs;

  /// MacroArgCache - This is a "freelist" of MacroArg objects that can be
  /// reused for quick allocation.
  MacroArgs *MacroArgCache;
  friend class MacroArgs;

  /// PragmaPushMacroInfo - For each IdentifierInfo used in a #pragma
  /// push_macro directive, we keep a MacroInfo stack used to restore
  /// previous macro value.
  llvm::DenseMap<IdentifierInfo*, std::vector<MacroInfo*> > PragmaPushMacroInfo;

  // Various statistics we track for performance analysis.
  unsigned NumDirectives, NumIncluded, NumDefined, NumUndefined, NumPragma;
  unsigned NumIf, NumElse, NumEndif;
  unsigned NumEnteredSourceFiles, MaxIncludeStackDepth;
  unsigned NumMacroExpanded, NumFnMacroExpanded, NumBuiltinMacroExpanded;
  unsigned NumFastMacroExpanded, NumTokenPaste, NumFastTokenPaste;
  unsigned NumSkipped;

  /// Predefines - This string is the predefined macros that preprocessor
  /// should use from the command line etc.
  std::string Predefines;

  /// TokenLexerCache - Cache macro expanders to reduce malloc traffic.
  enum { TokenLexerCacheSize = 8 };
  unsigned NumCachedTokenLexers;
  TokenLexer *TokenLexerCache[TokenLexerCacheSize];

  /// \brief Keeps macro expanded tokens for TokenLexers.
  //
  /// Works like a stack; a TokenLexer adds the macro expanded tokens that is
  /// going to lex in the cache and when it finishes the tokens are removed
  /// from the end of the cache.
  SmallVector<Token, 16> MacroExpandedTokens;
  std::vector<std::pair<TokenLexer *, size_t> > MacroExpandingLexersStack;

  /// \brief A record of the macro definitions and expansions that
  /// occurred during preprocessing.
  ///
  /// This is an optional side structure that can be enabled with
  /// \c createPreprocessingRecord() prior to preprocessing.
  PreprocessingRecord *Record;

private:  // Cached tokens state.
  typedef SmallVector<Token, 1> CachedTokensTy;

  /// CachedTokens - Cached tokens are stored here when we do backtracking or
  /// lookahead. They are "lexed" by the CachingLex() method.
  CachedTokensTy CachedTokens;

  /// CachedLexPos - The position of the cached token that CachingLex() should
  /// "lex" next. If it points beyond the CachedTokens vector, it means that
  /// a normal Lex() should be invoked.
  CachedTokensTy::size_type CachedLexPos;

  /// BacktrackPositions - Stack of backtrack positions, allowing nested
  /// backtracks. The EnableBacktrackAtThisPos() method pushes a position to
  /// indicate where CachedLexPos should be set when the BackTrack() method is
  /// invoked (at which point the last position is popped).
  std::vector<CachedTokensTy::size_type> BacktrackPositions;

  struct MacroInfoChain {
    MacroInfo MI;
    MacroInfoChain *Next;
    MacroInfoChain *Prev;
  };

  /// MacroInfos are managed as a chain for easy disposal.  This is the head
  /// of that list.
  MacroInfoChain *MIChainHead;

  /// MICache - A "freelist" of MacroInfo objects that can be reused for quick
  /// allocation.
  MacroInfoChain *MICache;

public:
  Preprocessor(llvm::IntrusiveRefCntPtr<PreprocessorOptions> PPOpts,
               DiagnosticsEngine &diags, LangOptions &opts,
               const TargetInfo *target,
               SourceManager &SM, HeaderSearch &Headers,
               ModuleLoader &TheModuleLoader,
               IdentifierInfoLookup *IILookup = 0,
               bool OwnsHeaderSearch = false,
               bool DelayInitialization = false,
               bool IncrProcessing = false);

  ~Preprocessor();

  /// \brief Initialize the preprocessor, if the constructor did not already
  /// perform the initialization.
  ///
  /// \param Target Information about the target.
  void Initialize(const TargetInfo &Target);

  /// \brief Retrieve the preprocessor options used to initialize this
  /// preprocessor.
  PreprocessorOptions &getPreprocessorOpts() const { return *PPOpts; }
  
  DiagnosticsEngine &getDiagnostics() const { return *Diags; }
  void setDiagnostics(DiagnosticsEngine &D) { Diags = &D; }

  const LangOptions &getLangOpts() const { return LangOpts; }
  const TargetInfo &getTargetInfo() const { return *Target; }
  FileManager &getFileManager() const { return FileMgr; }
  SourceManager &getSourceManager() const { return SourceMgr; }
  HeaderSearch &getHeaderSearchInfo() const { return HeaderInfo; }

  IdentifierTable &getIdentifierTable() { return Identifiers; }
  SelectorTable &getSelectorTable() { return Selectors; }
  Builtin::Context &getBuiltinInfo() { return BuiltinInfo; }
  llvm::BumpPtrAllocator &getPreprocessorAllocator() { return BP; }

  void setPTHManager(PTHManager* pm);

  PTHManager *getPTHManager() { return PTH.get(); }

  void setExternalSource(ExternalPreprocessorSource *Source) {
    ExternalSource = Source;
  }

  ExternalPreprocessorSource *getExternalSource() const {
    return ExternalSource;
  }

  /// \brief Retrieve the module loader associated with this preprocessor.
  ModuleLoader &getModuleLoader() const { return TheModuleLoader; }

  /// SetCommentRetentionState - Control whether or not the preprocessor retains
  /// comments in output.
  void SetCommentRetentionState(bool KeepComments, bool KeepMacroComments) {
    this->KeepComments = KeepComments | KeepMacroComments;
    this->KeepMacroComments = KeepMacroComments;
  }

  bool getCommentRetentionState() const { return KeepComments; }

  void setPragmasEnabled(bool Enabled) { PragmasEnabled = Enabled; }
  bool getPragmasEnabled() const { return PragmasEnabled; }

  void SetSuppressIncludeNotFoundError(bool Suppress) {
    SuppressIncludeNotFoundError = Suppress;
  }

  bool GetSuppressIncludeNotFoundError() {
    return SuppressIncludeNotFoundError;
  }

  /// isCurrentLexer - Return true if we are lexing directly from the specified
  /// lexer.
  bool isCurrentLexer(const PreprocessorLexer *L) const {
    return CurPPLexer == L;
  }

  /// getCurrentLexer - Return the current lexer being lexed from.  Note
  /// that this ignores any potentially active macro expansions and _Pragma
  /// expansions going on at the time.
  PreprocessorLexer *getCurrentLexer() const { return CurPPLexer; }

  /// getCurrentFileLexer - Return the current file lexer being lexed from.
  /// Note that this ignores any potentially active macro expansions and _Pragma
  /// expansions going on at the time.
  PreprocessorLexer *getCurrentFileLexer() const;

  /// getPPCallbacks/addPPCallbacks - Accessors for preprocessor callbacks.
  /// Note that this class takes ownership of any PPCallbacks object given to
  /// it.
  PPCallbacks *getPPCallbacks() const { return Callbacks; }
  void addPPCallbacks(PPCallbacks *C) {
    if (Callbacks)
      C = new PPChainedCallbacks(C, Callbacks);
    Callbacks = C;
  }

  /// \brief Attach an preprocessor mutation listener to the preprocessor.
  ///
  /// The preprocessor mutation listener provides the ability to track
  /// modifications to the preprocessor entities committed after they were
  /// initially created.
  void setPPMutationListener(PPMutationListener *Listener) {
    this->Listener = Listener;
  }

  /// \brief Retrieve a pointer to the preprocessor mutation listener
  /// associated with this preprocessor, if any.
  PPMutationListener *getPPMutationListener() const { return Listener; }

  /// \brief Given an identifier, return the MacroInfo it is \#defined to
  /// or null if it isn't \#define'd.
  MacroInfo *getMacroInfo(IdentifierInfo *II) const {
    if (!II->hasMacroDefinition())
      return 0;

    MacroInfo *MI = getMacroInfoHistory(II);
    assert(MI->getUndefLoc().isInvalid() && "Macro is undefined!");
    return MI;
  }

  /// \brief Given an identifier, return the (probably #undef'd) MacroInfo
  /// representing the most recent macro definition. One can iterate over all
  /// previous macro definitions from it. This method should only be called for
  /// identifiers that hadMacroDefinition().
  MacroInfo *getMacroInfoHistory(IdentifierInfo *II) const;

  /// \brief Specify a macro for this identifier.
  void setMacroInfo(IdentifierInfo *II, MacroInfo *MI);
  /// \brief Add a MacroInfo that was loaded from an AST file.
  void addLoadedMacroInfo(IdentifierInfo *II, MacroInfo *MI,
                          MacroInfo *Hint = 0);
  /// \brief Make the given MacroInfo, that was loaded from an AST file and
  /// previously hidden, visible.
  void makeLoadedMacroInfoVisible(IdentifierInfo *II, MacroInfo *MI);
  /// \brief Undefine a macro for this identifier.
  void clearMacroInfo(IdentifierInfo *II);

  /// macro_iterator/macro_begin/macro_end - This allows you to walk the macro
  /// history table. Currently defined macros have
  /// IdentifierInfo::hasMacroDefinition() set and an empty
  /// MacroInfo::getUndefLoc() at the head of the list.
  typedef llvm::DenseMap<IdentifierInfo*,
                         MacroInfo*>::const_iterator macro_iterator;
  macro_iterator macro_begin(bool IncludeExternalMacros = true) const;
  macro_iterator macro_end(bool IncludeExternalMacros = true) const;

  /// \brief Return the name of the macro defined before \p Loc that has
  /// spelling \p Tokens.  If there are multiple macros with same spelling,
  /// return the last one defined.
  StringRef getLastMacroWithSpelling(SourceLocation Loc,
                                     ArrayRef<TokenValue> Tokens) const;

  const std::string &getPredefines() const { return Predefines; }
  /// setPredefines - Set the predefines for this Preprocessor.  These
  /// predefines are automatically injected when parsing the main file.
  void setPredefines(const char *P) { Predefines = P; }
  void setPredefines(const std::string &P) { Predefines = P; }

  /// Return information about the specified preprocessor
  /// identifier token.
  IdentifierInfo *getIdentifierInfo(StringRef Name) const {
    return &Identifiers.get(Name);
  }

  /// AddPragmaHandler - Add the specified pragma handler to the preprocessor.
  /// If 'Namespace' is non-null, then it is a token required to exist on the
  /// pragma line before the pragma string starts, e.g. "STDC" or "GCC".
  void AddPragmaHandler(StringRef Namespace, PragmaHandler *Handler);
  void AddPragmaHandler(PragmaHandler *Handler) {
    AddPragmaHandler(StringRef(), Handler);
  }

  /// RemovePragmaHandler - Remove the specific pragma handler from
  /// the preprocessor. If \p Namespace is non-null, then it should
  /// be the namespace that \p Handler was added to. It is an error
  /// to remove a handler that has not been registered.
  void RemovePragmaHandler(StringRef Namespace, PragmaHandler *Handler);
  void RemovePragmaHandler(PragmaHandler *Handler) {
    RemovePragmaHandler(StringRef(), Handler);
  }

  /// \brief Add the specified comment handler to the preprocessor.
  void addCommentHandler(CommentHandler *Handler);

  /// \brief Remove the specified comment handler.
  ///
  /// It is an error to remove a handler that has not been registered.
  void removeCommentHandler(CommentHandler *Handler);

  /// \brief Set the code completion handler to the given object.
  void setCodeCompletionHandler(CodeCompletionHandler &Handler) {
    CodeComplete = &Handler;
  }

  /// \brief Retrieve the current code-completion handler.
  CodeCompletionHandler *getCodeCompletionHandler() const {
    return CodeComplete;
  }

  /// \brief Clear out the code completion handler.
  void clearCodeCompletionHandler() {
    CodeComplete = 0;
  }

  /// \brief Hook used by the lexer to invoke the "natural language" code
  /// completion point.
  void CodeCompleteNaturalLanguage();

  /// \brief Retrieve the preprocessing record, or NULL if there is no
  /// preprocessing record.
  PreprocessingRecord *getPreprocessingRecord() const { return Record; }

  /// \brief Create a new preprocessing record, which will keep track of
  /// all macro expansions, macro definitions, etc.
  void createPreprocessingRecord(bool RecordConditionalDirectives);

  /// EnterMainSourceFile - Enter the specified FileID as the main source file,
  /// which implicitly adds the builtin defines etc.
  void EnterMainSourceFile();

  /// EndSourceFile - Inform the preprocessor callbacks that processing is
  /// complete.
  void EndSourceFile();

  /// EnterSourceFile - Add a source file to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.  Emit an error
  /// and don't enter the file on error.
  void EnterSourceFile(FileID CurFileID, const DirectoryLookup *Dir,
                       SourceLocation Loc);

  /// EnterMacro - Add a Macro to the top of the include stack and start lexing
  /// tokens from it instead of the current buffer.  Args specifies the
  /// tokens input to a function-like macro.
  ///
  /// ILEnd specifies the location of the ')' for a function-like macro or the
  /// identifier for an object-like macro.
  void EnterMacro(Token &Identifier, SourceLocation ILEnd, MacroInfo *Macro,
                  MacroArgs *Args);

  /// EnterTokenStream - Add a "macro" context to the top of the include stack,
  /// which will cause the lexer to start returning the specified tokens.
  ///
  /// If DisableMacroExpansion is true, tokens lexed from the token stream will
  /// not be subject to further macro expansion.  Otherwise, these tokens will
  /// be re-macro-expanded when/if expansion is enabled.
  ///
  /// If OwnsTokens is false, this method assumes that the specified stream of
  /// tokens has a permanent owner somewhere, so they do not need to be copied.
  /// If it is true, it assumes the array of tokens is allocated with new[] and
  /// must be freed.
  ///
  void EnterTokenStream(const Token *Toks, unsigned NumToks,
                        bool DisableMacroExpansion, bool OwnsTokens);

  /// RemoveTopOfLexerStack - Pop the current lexer/macro exp off the top of the
  /// lexer stack.  This should only be used in situations where the current
  /// state of the top-of-stack lexer is known.
  void RemoveTopOfLexerStack();

  /// EnableBacktrackAtThisPos - From the point that this method is called, and
  /// until CommitBacktrackedTokens() or Backtrack() is called, the Preprocessor
  /// keeps track of the lexed tokens so that a subsequent Backtrack() call will
  /// make the Preprocessor re-lex the same tokens.
  ///
  /// Nested backtracks are allowed, meaning that EnableBacktrackAtThisPos can
  /// be called multiple times and CommitBacktrackedTokens/Backtrack calls will
  /// be combined with the EnableBacktrackAtThisPos calls in reverse order.
  ///
  /// NOTE: *DO NOT* forget to call either CommitBacktrackedTokens or Backtrack
  /// at some point after EnableBacktrackAtThisPos. If you don't, caching of
  /// tokens will continue indefinitely.
  ///
  void EnableBacktrackAtThisPos();

  /// CommitBacktrackedTokens - Disable the last EnableBacktrackAtThisPos call.
  void CommitBacktrackedTokens();

  /// Backtrack - Make Preprocessor re-lex the tokens that were lexed since
  /// EnableBacktrackAtThisPos() was previously called.
  void Backtrack();

  /// isBacktrackEnabled - True if EnableBacktrackAtThisPos() was called and
  /// caching of tokens is on.
  bool isBacktrackEnabled() const { return !BacktrackPositions.empty(); }

  /// Lex - To lex a token from the preprocessor, just pull a token from the
  /// current lexer or macro object.
  void Lex(Token &Result) {
    switch (CurLexerKind) {
    case CLK_Lexer: CurLexer->Lex(Result); break;
    case CLK_PTHLexer: CurPTHLexer->Lex(Result); break;
    case CLK_TokenLexer: CurTokenLexer->Lex(Result); break;
    case CLK_CachingLexer: CachingLex(Result); break;
    case CLK_LexAfterModuleImport: LexAfterModuleImport(Result); break;
    }
  }

  void LexAfterModuleImport(Token &Result);

  /// LexNonComment - Lex a token.  If it's a comment, keep lexing until we get
  /// something not a comment.  This is useful in -E -C mode where comments
  /// would foul up preprocessor directive handling.
  void LexNonComment(Token &Result) {
    do
      Lex(Result);
    while (Result.getKind() == tok::comment);
  }

  /// LexUnexpandedToken - This is just like Lex, but this disables macro
  /// expansion of identifier tokens.
  void LexUnexpandedToken(Token &Result) {
    // Disable macro expansion.
    bool OldVal = DisableMacroExpansion;
    DisableMacroExpansion = true;
    // Lex the token.
    Lex(Result);

    // Reenable it.
    DisableMacroExpansion = OldVal;
  }

  /// LexUnexpandedNonComment - Like LexNonComment, but this disables macro
  /// expansion of identifier tokens.
  void LexUnexpandedNonComment(Token &Result) {
    do
      LexUnexpandedToken(Result);
    while (Result.getKind() == tok::comment);
  }

  /// Disables macro expansion everywhere except for preprocessor directives.
  void SetMacroExpansionOnlyInDirectives() {
    DisableMacroExpansion = true;
    MacroExpansionInDirectivesOverride = true;
  }

  /// LookAhead - This peeks ahead N tokens and returns that token without
  /// consuming any tokens.  LookAhead(0) returns the next token that would be
  /// returned by Lex(), LookAhead(1) returns the token after it, etc.  This
  /// returns normal tokens after phase 5.  As such, it is equivalent to using
  /// 'Lex', not 'LexUnexpandedToken'.
  const Token &LookAhead(unsigned N) {
    if (CachedLexPos + N < CachedTokens.size())
      return CachedTokens[CachedLexPos+N];
    else
      return PeekAhead(N+1);
  }

  /// RevertCachedTokens - When backtracking is enabled and tokens are cached,
  /// this allows to revert a specific number of tokens.
  /// Note that the number of tokens being reverted should be up to the last
  /// backtrack position, not more.
  void RevertCachedTokens(unsigned N) {
    assert(isBacktrackEnabled() &&
           "Should only be called when tokens are cached for backtracking");
    assert(signed(CachedLexPos) - signed(N) >= signed(BacktrackPositions.back())
         && "Should revert tokens up to the last backtrack position, not more");
    assert(signed(CachedLexPos) - signed(N) >= 0 &&
           "Corrupted backtrack positions ?");
    CachedLexPos -= N;
  }

  /// EnterToken - Enters a token in the token stream to be lexed next. If
  /// BackTrack() is called afterwards, the token will remain at the insertion
  /// point.
  void EnterToken(const Token &Tok) {
    EnterCachingLexMode();
    CachedTokens.insert(CachedTokens.begin()+CachedLexPos, Tok);
  }

  /// AnnotateCachedTokens - We notify the Preprocessor that if it is caching
  /// tokens (because backtrack is enabled) it should replace the most recent
  /// cached tokens with the given annotation token. This function has no effect
  /// if backtracking is not enabled.
  ///
  /// Note that the use of this function is just for optimization; so that the
  /// cached tokens doesn't get re-parsed and re-resolved after a backtrack is
  /// invoked.
  void AnnotateCachedTokens(const Token &Tok) {
    assert(Tok.isAnnotation() && "Expected annotation token");
    if (CachedLexPos != 0 && isBacktrackEnabled())
      AnnotatePreviousCachedTokens(Tok);
  }

  /// \brief Replace the last token with an annotation token.
  ///
  /// Like AnnotateCachedTokens(), this routine replaces an
  /// already-parsed (and resolved) token with an annotation
  /// token. However, this routine only replaces the last token with
  /// the annotation token; it does not affect any other cached
  /// tokens. This function has no effect if backtracking is not
  /// enabled.
  void ReplaceLastTokenWithAnnotation(const Token &Tok) {
    assert(Tok.isAnnotation() && "Expected annotation token");
    if (CachedLexPos != 0 && isBacktrackEnabled())
      CachedTokens[CachedLexPos-1] = Tok;
  }

  /// TypoCorrectToken - Update the current token to represent the provided
  /// identifier, in order to cache an action performed by typo correction.
  void TypoCorrectToken(const Token &Tok) {
    assert(Tok.getIdentifierInfo() && "Expected identifier token");
    if (CachedLexPos != 0 && isBacktrackEnabled())
      CachedTokens[CachedLexPos-1] = Tok;
  }

  /// \brief Recompute the current lexer kind based on the CurLexer/CurPTHLexer/
  /// CurTokenLexer pointers.
  void recomputeCurLexerKind();

  /// \brief Returns true if incremental processing is enabled
  bool isIncrementalProcessingEnabled() const { return IncrementalProcessing; }

  /// \brief Enables the incremental processing
  void enableIncrementalProcessing(bool value = true) {
    IncrementalProcessing = value;
  }
  
  /// \brief Specify the point at which code-completion will be performed.
  ///
  /// \param File the file in which code completion should occur. If
  /// this file is included multiple times, code-completion will
  /// perform completion the first time it is included. If NULL, this
  /// function clears out the code-completion point.
  ///
  /// \param Line the line at which code completion should occur
  /// (1-based).
  ///
  /// \param Column the column at which code completion should occur
  /// (1-based).
  ///
  /// \returns true if an error occurred, false otherwise.
  bool SetCodeCompletionPoint(const FileEntry *File,
                              unsigned Line, unsigned Column);

  /// \brief Determine if we are performing code completion.
  bool isCodeCompletionEnabled() const { return CodeCompletionFile != 0; }

  /// \brief Returns the location of the code-completion point.
  /// Returns an invalid location if code-completion is not enabled or the file
  /// containing the code-completion point has not been lexed yet.
  SourceLocation getCodeCompletionLoc() const { return CodeCompletionLoc; }

  /// \brief Returns the start location of the file of code-completion point.
  /// Returns an invalid location if code-completion is not enabled or the file
  /// containing the code-completion point has not been lexed yet.
  SourceLocation getCodeCompletionFileLoc() const {
    return CodeCompletionFileLoc;
  }

  /// \brief Returns true if code-completion is enabled and we have hit the
  /// code-completion point.
  bool isCodeCompletionReached() const { return CodeCompletionReached; }

  /// \brief Note that we hit the code-completion point.
  void setCodeCompletionReached() {
    assert(isCodeCompletionEnabled() && "Code-completion not enabled!");
    CodeCompletionReached = true;
    // Silence any diagnostics that occur after we hit the code-completion.
    getDiagnostics().setSuppressAllDiagnostics(true);
  }

  /// \brief The location of the currently-active \#pragma clang
  /// arc_cf_code_audited begin.  Returns an invalid location if there
  /// is no such pragma active.
  SourceLocation getPragmaARCCFCodeAuditedLoc() const {
    return PragmaARCCFCodeAuditedLoc;
  }

  /// \brief Set the location of the currently-active \#pragma clang
  /// arc_cf_code_audited begin.  An invalid location ends the pragma.
  void setPragmaARCCFCodeAuditedLoc(SourceLocation Loc) {
    PragmaARCCFCodeAuditedLoc = Loc;
  }

  /// \brief Instruct the preprocessor to skip part of the main source file.
  ///
  /// \param Bytes The number of bytes in the preamble to skip.
  ///
  /// \param StartOfLine Whether skipping these bytes puts the lexer at the
  /// start of a line.
  void setSkipMainFilePreamble(unsigned Bytes, bool StartOfLine) {
    SkipMainFilePreamble.first = Bytes;
    SkipMainFilePreamble.second = StartOfLine;
  }

  /// Diag - Forwarding function for diagnostics.  This emits a diagnostic at
  /// the specified Token's location, translating the token's start
  /// position in the current buffer into a SourcePosition object for rendering.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) const {
    return Diags->Report(Loc, DiagID);
  }

  DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID) const {
    return Diags->Report(Tok.getLocation(), DiagID);
  }

  /// getSpelling() - Return the 'spelling' of the token at the given
  /// location; does not go up to the spelling location or down to the
  /// expansion location.
  ///
  /// \param buffer A buffer which will be used only if the token requires
  ///   "cleaning", e.g. if it contains trigraphs or escaped newlines
  /// \param invalid If non-null, will be set \c true if an error occurs.
  StringRef getSpelling(SourceLocation loc,
                              SmallVectorImpl<char> &buffer,
                              bool *invalid = 0) const {
    return Lexer::getSpelling(loc, buffer, SourceMgr, LangOpts, invalid);
  }

  /// getSpelling() - Return the 'spelling' of the Tok token.  The spelling of a
  /// token is the characters used to represent the token in the source file
  /// after trigraph expansion and escaped-newline folding.  In particular, this
  /// wants to get the true, uncanonicalized, spelling of things like digraphs
  /// UCNs, etc.
  ///
  /// \param Invalid If non-null, will be set \c true if an error occurs.
  std::string getSpelling(const Token &Tok, bool *Invalid = 0) const {
    return Lexer::getSpelling(Tok, SourceMgr, LangOpts, Invalid);
  }

  /// getSpelling - This method is used to get the spelling of a token into a
  /// preallocated buffer, instead of as an std::string.  The caller is required
  /// to allocate enough space for the token, which is guaranteed to be at least
  /// Tok.getLength() bytes long.  The length of the actual result is returned.
  ///
  /// Note that this method may do two possible things: it may either fill in
  /// the buffer specified with characters, or it may *change the input pointer*
  /// to point to a constant buffer with the data already in it (avoiding a
  /// copy).  The caller is not allowed to modify the returned buffer pointer
  /// if an internal buffer is returned.
  unsigned getSpelling(const Token &Tok, const char *&Buffer,
                       bool *Invalid = 0) const {
    return Lexer::getSpelling(Tok, Buffer, SourceMgr, LangOpts, Invalid);
  }

  /// getSpelling - This method is used to get the spelling of a token into a
  /// SmallVector. Note that the returned StringRef may not point to the
  /// supplied buffer if a copy can be avoided.
  StringRef getSpelling(const Token &Tok,
                        SmallVectorImpl<char> &Buffer,
                        bool *Invalid = 0) const;

  /// getSpellingOfSingleCharacterNumericConstant - Tok is a numeric constant
  /// with length 1, return the character.
  char getSpellingOfSingleCharacterNumericConstant(const Token &Tok,
                                                   bool *Invalid = 0) const {
    assert(Tok.is(tok::numeric_constant) &&
           Tok.getLength() == 1 && "Called on unsupported token");
    assert(!Tok.needsCleaning() && "Token can't need cleaning with length 1");

    // If the token is carrying a literal data pointer, just use it.
    if (const char *D = Tok.getLiteralData())
      return *D;

    // Otherwise, fall back on getCharacterData, which is slower, but always
    // works.
    return *SourceMgr.getCharacterData(Tok.getLocation(), Invalid);
  }

  /// \brief Retrieve the name of the immediate macro expansion.
  ///
  /// This routine starts from a source location, and finds the name of the macro
  /// responsible for its immediate expansion. It looks through any intervening
  /// macro argument expansions to compute this. It returns a StringRef which
  /// refers to the SourceManager-owned buffer of the source where that macro
  /// name is spelled. Thus, the result shouldn't out-live the SourceManager.
  StringRef getImmediateMacroName(SourceLocation Loc) {
    return Lexer::getImmediateMacroName(Loc, SourceMgr, getLangOpts());
  }

  /// CreateString - Plop the specified string into a scratch buffer and set the
  /// specified token's location and length to it.  If specified, the source
  /// location provides a location of the expansion point of the token.
  void CreateString(StringRef Str, Token &Tok,
                    SourceLocation ExpansionLocStart = SourceLocation(),
                    SourceLocation ExpansionLocEnd = SourceLocation());

  /// \brief Computes the source location just past the end of the
  /// token at this source location.
  ///
  /// This routine can be used to produce a source location that
  /// points just past the end of the token referenced by \p Loc, and
  /// is generally used when a diagnostic needs to point just after a
  /// token where it expected something different that it received. If
  /// the returned source location would not be meaningful (e.g., if
  /// it points into a macro), this routine returns an invalid
  /// source location.
  ///
  /// \param Offset an offset from the end of the token, where the source
  /// location should refer to. The default offset (0) produces a source
  /// location pointing just past the end of the token; an offset of 1 produces
  /// a source location pointing to the last character in the token, etc.
  SourceLocation getLocForEndOfToken(SourceLocation Loc, unsigned Offset = 0) {
    return Lexer::getLocForEndOfToken(Loc, Offset, SourceMgr, LangOpts);
  }

  /// \brief Returns true if the given MacroID location points at the first
  /// token of the macro expansion.
  ///
  /// \param MacroBegin If non-null and function returns true, it is set to
  /// begin location of the macro.
  bool isAtStartOfMacroExpansion(SourceLocation loc,
                                 SourceLocation *MacroBegin = 0) const {
    return Lexer::isAtStartOfMacroExpansion(loc, SourceMgr, LangOpts,
                                            MacroBegin);
  }

  /// \brief Returns true if the given MacroID location points at the last
  /// token of the macro expansion.
  ///
  /// \param MacroEnd If non-null and function returns true, it is set to
  /// end location of the macro.
  bool isAtEndOfMacroExpansion(SourceLocation loc,
                               SourceLocation *MacroEnd = 0) const {
    return Lexer::isAtEndOfMacroExpansion(loc, SourceMgr, LangOpts, MacroEnd);
  }

  /// DumpToken - Print the token to stderr, used for debugging.
  ///
  void DumpToken(const Token &Tok, bool DumpFlags = false) const;
  void DumpLocation(SourceLocation Loc) const;
  void DumpMacro(const MacroInfo &MI) const;

  /// AdvanceToTokenCharacter - Given a location that specifies the start of a
  /// token, return a new location that specifies a character within the token.
  SourceLocation AdvanceToTokenCharacter(SourceLocation TokStart,
                                         unsigned Char) const {
    return Lexer::AdvanceToTokenCharacter(TokStart, Char, SourceMgr, LangOpts);
  }

  /// IncrementPasteCounter - Increment the counters for the number of token
  /// paste operations performed.  If fast was specified, this is a 'fast paste'
  /// case we handled.
  ///
  void IncrementPasteCounter(bool isFast) {
    if (isFast)
      ++NumFastTokenPaste;
    else
      ++NumTokenPaste;
  }

  void PrintStats();

  size_t getTotalMemory() const;

  /// HandleMicrosoftCommentPaste - When the macro expander pastes together a
  /// comment (/##/) in microsoft mode, this method handles updating the current
  /// state, returning the token on the next source line.
  void HandleMicrosoftCommentPaste(Token &Tok);

  //===--------------------------------------------------------------------===//
  // Preprocessor callback methods.  These are invoked by a lexer as various
  // directives and events are found.

  /// LookUpIdentifierInfo - Given a tok::raw_identifier token, look up the
  /// identifier information for the token and install it into the token,
  /// updating the token kind accordingly.
  IdentifierInfo *LookUpIdentifierInfo(Token &Identifier) const;

private:
  llvm::DenseMap<IdentifierInfo*,unsigned> PoisonReasons;

public:

  // SetPoisonReason - Call this function to indicate the reason for
  // poisoning an identifier. If that identifier is accessed while
  // poisoned, then this reason will be used instead of the default
  // "poisoned" diagnostic.
  void SetPoisonReason(IdentifierInfo *II, unsigned DiagID);

  // HandlePoisonedIdentifier - Display reason for poisoned
  // identifier.
  void HandlePoisonedIdentifier(Token & Tok);

  void MaybeHandlePoisonedIdentifier(Token & Identifier) {
    if(IdentifierInfo * II = Identifier.getIdentifierInfo()) {
      if(II->isPoisoned()) {
        HandlePoisonedIdentifier(Identifier);
      }
    }
  }

private:
  /// Identifiers used for SEH handling in Borland. These are only
  /// allowed in particular circumstances
  // __except block
  IdentifierInfo *Ident__exception_code,
                 *Ident___exception_code,
                 *Ident_GetExceptionCode;
  // __except filter expression
  IdentifierInfo *Ident__exception_info,
                 *Ident___exception_info,
                 *Ident_GetExceptionInfo;
  // __finally
  IdentifierInfo *Ident__abnormal_termination,
                 *Ident___abnormal_termination,
                 *Ident_AbnormalTermination;
public:
  void PoisonSEHIdentifiers(bool Poison = true); // Borland

  /// HandleIdentifier - This callback is invoked when the lexer reads an
  /// identifier and has filled in the tokens IdentifierInfo member.  This
  /// callback potentially macro expands it or turns it into a named token (like
  /// 'for').
  void HandleIdentifier(Token &Identifier);


  /// HandleEndOfFile - This callback is invoked when the lexer hits the end of
  /// the current file.  This either returns the EOF token and returns true, or
  /// pops a level off the include stack and returns false, at which point the
  /// client should call lex again.
  bool HandleEndOfFile(Token &Result, bool isEndOfMacro = false);

  /// HandleEndOfTokenLexer - This callback is invoked when the current
  /// TokenLexer hits the end of its token stream.
  bool HandleEndOfTokenLexer(Token &Result);

  /// HandleDirective - This callback is invoked when the lexer sees a # token
  /// at the start of a line.  This consumes the directive, modifies the
  /// lexer/preprocessor state, and advances the lexer(s) so that the next token
  /// read is the correct one.
  void HandleDirective(Token &Result);

  /// CheckEndOfDirective - Ensure that the next token is a tok::eod token.  If
  /// not, emit a diagnostic and consume up until the eod.  If EnableMacros is
  /// true, then we consider macros that expand to zero tokens as being ok.
  void CheckEndOfDirective(const char *Directive, bool EnableMacros = false);

  /// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
  /// current line until the tok::eod token is found.
  void DiscardUntilEndOfDirective();

  /// SawDateOrTime - This returns true if the preprocessor has seen a use of
  /// __DATE__ or __TIME__ in the file so far.
  bool SawDateOrTime() const {
    return DATELoc != SourceLocation() || TIMELoc != SourceLocation();
  }
  unsigned getCounterValue() const { return CounterValue; }
  void setCounterValue(unsigned V) { CounterValue = V; }

  /// \brief Retrieves the module that we're currently building, if any.
  Module *getCurrentModule();
  
  /// \brief Allocate a new MacroInfo object with the provided SourceLocation.
  MacroInfo *AllocateMacroInfo(SourceLocation L);

  /// \brief Allocate a new MacroInfo object which is clone of \p MI.
  MacroInfo *CloneMacroInfo(const MacroInfo &MI);

  /// \brief Turn the specified lexer token into a fully checked and spelled
  /// filename, e.g. as an operand of \#include. 
  ///
  /// The caller is expected to provide a buffer that is large enough to hold
  /// the spelling of the filename, but is also expected to handle the case
  /// when this method decides to use a different buffer.
  ///
  /// \returns true if the input filename was in <>'s or false if it was
  /// in ""'s.
  bool GetIncludeFilenameSpelling(SourceLocation Loc,StringRef &Filename);

  /// \brief Given a "foo" or \<foo> reference, look up the indicated file.
  ///
  /// Returns null on failure.  \p isAngled indicates whether the file
  /// reference is for system \#include's or not (i.e. using <> instead of "").
  const FileEntry *LookupFile(StringRef Filename,
                              bool isAngled, const DirectoryLookup *FromDir,
                              const DirectoryLookup *&CurDir,
                              SmallVectorImpl<char> *SearchPath,
                              SmallVectorImpl<char> *RelativePath,
                              Module **SuggestedModule,
                              bool SkipCache = false);

  /// GetCurLookup - The DirectoryLookup structure used to find the current
  /// FileEntry, if CurLexer is non-null and if applicable.  This allows us to
  /// implement \#include_next and find directory-specific properties.
  const DirectoryLookup *GetCurDirLookup() { return CurDirLookup; }

  /// \brief Return true if we're in the top-level file, not in a \#include.
  bool isInPrimaryFile() const;

  /// ConcatenateIncludeName - Handle cases where the \#include name is expanded
  /// from a macro as multiple tokens, which need to be glued together.  This
  /// occurs for code like:
  /// \code
  ///    \#define FOO <x/y.h>
  ///    \#include FOO
  /// \endcode
  /// because in this case, "<x/y.h>" is returned as 7 tokens, not one.
  ///
  /// This code concatenates and consumes tokens up to the '>' token.  It
  /// returns false if the > was found, otherwise it returns true if it finds
  /// and consumes the EOD marker.
  bool ConcatenateIncludeName(SmallString<128> &FilenameBuffer,
                              SourceLocation &End);

  /// LexOnOffSwitch - Lex an on-off-switch (C99 6.10.6p2) and verify that it is
  /// followed by EOD.  Return true if the token is not a valid on-off-switch.
  bool LexOnOffSwitch(tok::OnOffSwitch &OOS);

private:

  void PushIncludeMacroStack() {
    IncludeMacroStack.push_back(IncludeStackInfo(CurLexerKind,
                                                 CurLexer.take(),
                                                 CurPTHLexer.take(),
                                                 CurPPLexer,
                                                 CurTokenLexer.take(),
                                                 CurDirLookup));
    CurPPLexer = 0;
  }

  void PopIncludeMacroStack() {
    CurLexer.reset(IncludeMacroStack.back().TheLexer);
    CurPTHLexer.reset(IncludeMacroStack.back().ThePTHLexer);
    CurPPLexer = IncludeMacroStack.back().ThePPLexer;
    CurTokenLexer.reset(IncludeMacroStack.back().TheTokenLexer);
    CurDirLookup  = IncludeMacroStack.back().TheDirLookup;
    CurLexerKind = IncludeMacroStack.back().CurLexerKind;
    IncludeMacroStack.pop_back();
  }

  /// \brief Allocate a new MacroInfo object.
  MacroInfo *AllocateMacroInfo();

  /// \brief Release the specified MacroInfo for re-use.
  ///
  /// This memory will  be reused for allocating new MacroInfo objects.
  void ReleaseMacroInfo(MacroInfo* MI);

  /// ReadMacroName - Lex and validate a macro name, which occurs after a
  /// \#define or \#undef.  This emits a diagnostic, sets the token kind to eod,
  /// and discards the rest of the macro line if the macro name is invalid.
  void ReadMacroName(Token &MacroNameTok, char isDefineUndef = 0);

  /// ReadMacroDefinitionArgList - The ( starting an argument list of a macro
  /// definition has just been read.  Lex the rest of the arguments and the
  /// closing ), updating MI with what we learn and saving in LastTok the
  /// last token read.
  /// Return true if an error occurs parsing the arg list.
  bool ReadMacroDefinitionArgList(MacroInfo *MI, Token& LastTok);

  /// We just read a \#if or related directive and decided that the
  /// subsequent tokens are in the \#if'd out portion of the
  /// file.  Lex the rest of the file, until we see an \#endif.  If \p
  /// FoundNonSkipPortion is true, then we have already emitted code for part of
  /// this \#if directive, so \#else/\#elif blocks should never be entered. If
  /// \p FoundElse is false, then \#else directives are ok, if not, then we have
  /// already seen one so a \#else directive is a duplicate.  When this returns,
  /// the caller can lex the first valid token.
  void SkipExcludedConditionalBlock(SourceLocation IfTokenLoc,
                                    bool FoundNonSkipPortion, bool FoundElse,
                                    SourceLocation ElseLoc = SourceLocation());

  /// \brief A fast PTH version of SkipExcludedConditionalBlock.
  void PTHSkipExcludedConditionalBlock();

  /// EvaluateDirectiveExpression - Evaluate an integer constant expression that
  /// may occur after a #if or #elif directive and return it as a bool.  If the
  /// expression is equivalent to "!defined(X)" return X in IfNDefMacro.
  bool EvaluateDirectiveExpression(IdentifierInfo *&IfNDefMacro);

  /// RegisterBuiltinPragmas - Install the standard preprocessor pragmas:
  /// \#pragma GCC poison/system_header/dependency and \#pragma once.
  void RegisterBuiltinPragmas();

  /// \brief Register builtin macros such as __LINE__ with the identifier table.
  void RegisterBuiltinMacros();

  /// HandleMacroExpandedIdentifier - If an identifier token is read that is to
  /// be expanded as a macro, handle it and return the next token as 'Tok'.  If
  /// the macro should not be expanded return true, otherwise return false.
  bool HandleMacroExpandedIdentifier(Token &Tok, MacroInfo *MI);

  /// \brief Cache macro expanded tokens for TokenLexers.
  //
  /// Works like a stack; a TokenLexer adds the macro expanded tokens that is
  /// going to lex in the cache and when it finishes the tokens are removed
  /// from the end of the cache.
  Token *cacheMacroExpandedTokens(TokenLexer *tokLexer,
                                  ArrayRef<Token> tokens);
  void removeCachedMacroExpandedTokensOfLastLexer();
  friend void TokenLexer::ExpandFunctionArguments();

  /// isNextPPTokenLParen - Determine whether the next preprocessor token to be
  /// lexed is a '('.  If so, consume the token and return true, if not, this
  /// method should have no observable side-effect on the lexed tokens.
  bool isNextPPTokenLParen();

  /// ReadFunctionLikeMacroArgs - After reading "MACRO(", this method is
  /// invoked to read all of the formal arguments specified for the macro
  /// invocation.  This returns null on error.
  MacroArgs *ReadFunctionLikeMacroArgs(Token &MacroName, MacroInfo *MI,
                                       SourceLocation &ExpansionEnd);

  /// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
  /// as a builtin macro, handle it and return the next token as 'Tok'.
  void ExpandBuiltinMacro(Token &Tok);

  /// Handle_Pragma - Read a _Pragma directive, slice it up, process it, then
  /// return the first token after the directive.  The _Pragma token has just
  /// been read into 'Tok'.
  void Handle_Pragma(Token &Tok);

  /// HandleMicrosoft__pragma - Like Handle_Pragma except the pragma text
  /// is not enclosed within a string literal.
  void HandleMicrosoft__pragma(Token &Tok);

  /// EnterSourceFileWithLexer - Add a lexer to the top of the include stack and
  /// start lexing tokens from it instead of the current buffer.
  void EnterSourceFileWithLexer(Lexer *TheLexer, const DirectoryLookup *Dir);

  /// EnterSourceFileWithPTH - Add a lexer to the top of the include stack and
  /// start getting tokens from it using the PTH cache.
  void EnterSourceFileWithPTH(PTHLexer *PL, const DirectoryLookup *Dir);

  /// IsFileLexer - Returns true if we are lexing from a file and not a
  ///  pragma or a macro.
  static bool IsFileLexer(const Lexer* L, const PreprocessorLexer* P) {
    return L ? !L->isPragmaLexer() : P != 0;
  }

  static bool IsFileLexer(const IncludeStackInfo& I) {
    return IsFileLexer(I.TheLexer, I.ThePPLexer);
  }

  bool IsFileLexer() const {
    return IsFileLexer(CurLexer.get(), CurPPLexer);
  }

  //===--------------------------------------------------------------------===//
  // Caching stuff.
  void CachingLex(Token &Result);
  bool InCachingLexMode() const {
    // If the Lexer pointers are 0 and IncludeMacroStack is empty, it means
    // that we are past EOF, not that we are in CachingLex mode.
    return CurPPLexer == 0 && CurTokenLexer == 0 && CurPTHLexer == 0 &&
           !IncludeMacroStack.empty();
  }
  void EnterCachingLexMode();
  void ExitCachingLexMode() {
    if (InCachingLexMode())
      RemoveTopOfLexerStack();
  }
  const Token &PeekAhead(unsigned N);
  void AnnotatePreviousCachedTokens(const Token &Tok);

  //===--------------------------------------------------------------------===//
  /// Handle*Directive - implement the various preprocessor directives.  These
  /// should side-effect the current preprocessor object so that the next call
  /// to Lex() will return the appropriate token next.
  void HandleLineDirective(Token &Tok);
  void HandleDigitDirective(Token &Tok);
  void HandleUserDiagnosticDirective(Token &Tok, bool isWarning);
  void HandleIdentSCCSDirective(Token &Tok);
  void HandleMacroPublicDirective(Token &Tok);
  void HandleMacroPrivateDirective(Token &Tok);

  // File inclusion.
  void HandleIncludeDirective(SourceLocation HashLoc,
                              Token &Tok,
                              const DirectoryLookup *LookupFrom = 0,
                              bool isImport = false);
  void HandleIncludeNextDirective(SourceLocation HashLoc, Token &Tok);
  void HandleIncludeMacrosDirective(SourceLocation HashLoc, Token &Tok);
  void HandleImportDirective(SourceLocation HashLoc, Token &Tok);
  void HandleMicrosoftImportDirective(Token &Tok);

  // Macro handling.
  void HandleDefineDirective(Token &Tok);
  void HandleUndefDirective(Token &Tok);
  void UndefineMacro(IdentifierInfo *II, MacroInfo *MI,
                     SourceLocation UndefLoc);

  // Conditional Inclusion.
  void HandleIfdefDirective(Token &Tok, bool isIfndef,
                            bool ReadAnyTokensBeforeDirective);
  void HandleIfDirective(Token &Tok, bool ReadAnyTokensBeforeDirective);
  void HandleEndifDirective(Token &Tok);
  void HandleElseDirective(Token &Tok);
  void HandleElifDirective(Token &Tok);

  // Pragmas.
  void HandlePragmaDirective(unsigned Introducer);
public:
  void HandlePragmaOnce(Token &OnceTok);
  void HandlePragmaMark();
  void HandlePragmaPoison(Token &PoisonTok);
  void HandlePragmaSystemHeader(Token &SysHeaderTok);
  void HandlePragmaDependency(Token &DependencyTok);
  void HandlePragmaComment(Token &CommentTok);
  void HandlePragmaMessage(Token &MessageTok);
  void HandlePragmaPushMacro(Token &Tok);
  void HandlePragmaPopMacro(Token &Tok);
  void HandlePragmaIncludeAlias(Token &Tok);
  IdentifierInfo *ParsePragmaPushOrPopMacro(Token &Tok);

  // Return true and store the first token only if any CommentHandler
  // has inserted some tokens and getCommentRetentionState() is false.
  bool HandleComment(Token &Token, SourceRange Comment);

  /// \brief A macro is used, update information about macros that need unused
  /// warnings.
  void markMacroAsUsed(MacroInfo *MI);
};

/// \brief Abstract base class that describes a handler that will receive
/// source ranges for each of the comments encountered in the source file.
class CommentHandler {
public:
  virtual ~CommentHandler();

  // The handler shall return true if it has pushed any tokens
  // to be read using e.g. EnterToken or EnterTokenStream.
  virtual bool HandleComment(Preprocessor &PP, SourceRange Comment) = 0;
};

}  // end namespace clang

#endif
