//===--- MacroExpansion.cpp - Top level Macro Expansion -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the top level handling of macro expasion for the
// preprocessor.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/MacroInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <ctime>
using namespace clang;

MacroDirective *
Preprocessor::getMacroDirectiveHistory(const IdentifierInfo *II) const {
  assert(II->hadMacroDefinition() && "Identifier has not been not a macro!");

  macro_iterator Pos = Macros.find(II);
  assert(Pos != Macros.end() && "Identifier macro info is missing!");
  return Pos->second;
}

void Preprocessor::appendMacroDirective(IdentifierInfo *II, MacroDirective *MD){
  assert(MD && "MacroDirective should be non-zero!");
  assert(!MD->getPrevious() && "Already attached to a MacroDirective history.");

  MacroDirective *&StoredMD = Macros[II];
  MD->setPrevious(StoredMD);
  StoredMD = MD;
  II->setHasMacroDefinition(MD->isDefined());
  bool isImportedMacro = isa<DefMacroDirective>(MD) &&
                         cast<DefMacroDirective>(MD)->isImported();
  if (II->isFromAST() && !isImportedMacro)
    II->setChangedSinceDeserialization();
}

void Preprocessor::setLoadedMacroDirective(IdentifierInfo *II,
                                           MacroDirective *MD) {
  assert(II && MD);
  MacroDirective *&StoredMD = Macros[II];
  assert(!StoredMD &&
         "the macro history was modified before initializing it from a pch");
  StoredMD = MD;
  // Setup the identifier as having associated macro history.
  II->setHasMacroDefinition(true);
  if (!MD->isDefined())
    II->setHasMacroDefinition(false);
}

/// RegisterBuiltinMacro - Register the specified identifier in the identifier
/// table and mark it as a builtin macro to be expanded.
static IdentifierInfo *RegisterBuiltinMacro(Preprocessor &PP, const char *Name){
  // Get the identifier.
  IdentifierInfo *Id = PP.getIdentifierInfo(Name);

  // Mark it as being a macro that is builtin.
  MacroInfo *MI = PP.AllocateMacroInfo(SourceLocation());
  MI->setIsBuiltinMacro();
  PP.appendDefMacroDirective(Id, MI);
  return Id;
}


/// RegisterBuiltinMacros - Register builtin macros, such as __LINE__ with the
/// identifier table.
void Preprocessor::RegisterBuiltinMacros() {
  Ident__LINE__ = RegisterBuiltinMacro(*this, "__LINE__");
  Ident__FILE__ = RegisterBuiltinMacro(*this, "__FILE__");
  Ident__DATE__ = RegisterBuiltinMacro(*this, "__DATE__");
  Ident__TIME__ = RegisterBuiltinMacro(*this, "__TIME__");
  Ident__COUNTER__ = RegisterBuiltinMacro(*this, "__COUNTER__");
  Ident_Pragma  = RegisterBuiltinMacro(*this, "_Pragma");

  // GCC Extensions.
  Ident__BASE_FILE__     = RegisterBuiltinMacro(*this, "__BASE_FILE__");
  Ident__INCLUDE_LEVEL__ = RegisterBuiltinMacro(*this, "__INCLUDE_LEVEL__");
  Ident__TIMESTAMP__     = RegisterBuiltinMacro(*this, "__TIMESTAMP__");

  // Clang Extensions.
  Ident__has_feature      = RegisterBuiltinMacro(*this, "__has_feature");
  Ident__has_extension    = RegisterBuiltinMacro(*this, "__has_extension");
  Ident__has_builtin      = RegisterBuiltinMacro(*this, "__has_builtin");
  Ident__has_attribute    = RegisterBuiltinMacro(*this, "__has_attribute");
  Ident__has_include      = RegisterBuiltinMacro(*this, "__has_include");
  Ident__has_include_next = RegisterBuiltinMacro(*this, "__has_include_next");
  Ident__has_warning      = RegisterBuiltinMacro(*this, "__has_warning");

  // Modules.
  if (LangOpts.Modules) {
    Ident__building_module  = RegisterBuiltinMacro(*this, "__building_module");

    // __MODULE__
    if (!LangOpts.CurrentModule.empty())
      Ident__MODULE__ = RegisterBuiltinMacro(*this, "__MODULE__");
    else
      Ident__MODULE__ = 0;
  } else {
    Ident__building_module = 0;
    Ident__MODULE__ = 0;
  }
  
  // Microsoft Extensions.
  if (LangOpts.MicrosoftExt) 
    Ident__pragma = RegisterBuiltinMacro(*this, "__pragma");
  else
    Ident__pragma = 0;
}

/// isTrivialSingleTokenExpansion - Return true if MI, which has a single token
/// in its expansion, currently expands to that token literally.
static bool isTrivialSingleTokenExpansion(const MacroInfo *MI,
                                          const IdentifierInfo *MacroIdent,
                                          Preprocessor &PP) {
  IdentifierInfo *II = MI->getReplacementToken(0).getIdentifierInfo();

  // If the token isn't an identifier, it's always literally expanded.
  if (II == 0) return true;

  // If the information about this identifier is out of date, update it from
  // the external source.
  if (II->isOutOfDate())
    PP.getExternalSource()->updateOutOfDateIdentifier(*II);

  // If the identifier is a macro, and if that macro is enabled, it may be
  // expanded so it's not a trivial expansion.
  if (II->hasMacroDefinition() && PP.getMacroInfo(II)->isEnabled() &&
      // Fast expanding "#define X X" is ok, because X would be disabled.
      II != MacroIdent)
    return false;

  // If this is an object-like macro invocation, it is safe to trivially expand
  // it.
  if (MI->isObjectLike()) return true;

  // If this is a function-like macro invocation, it's safe to trivially expand
  // as long as the identifier is not a macro argument.
  for (MacroInfo::arg_iterator I = MI->arg_begin(), E = MI->arg_end();
       I != E; ++I)
    if (*I == II)
      return false;   // Identifier is a macro argument.

  return true;
}


/// isNextPPTokenLParen - Determine whether the next preprocessor token to be
/// lexed is a '('.  If so, consume the token and return true, if not, this
/// method should have no observable side-effect on the lexed tokens.
bool Preprocessor::isNextPPTokenLParen() {
  // Do some quick tests for rejection cases.
  unsigned Val;
  if (CurLexer)
    Val = CurLexer->isNextPPTokenLParen();
  else if (CurPTHLexer)
    Val = CurPTHLexer->isNextPPTokenLParen();
  else
    Val = CurTokenLexer->isNextTokenLParen();

  if (Val == 2) {
    // We have run off the end.  If it's a source file we don't
    // examine enclosing ones (C99 5.1.1.2p4).  Otherwise walk up the
    // macro stack.
    if (CurPPLexer)
      return false;
    for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
      IncludeStackInfo &Entry = IncludeMacroStack[i-1];
      if (Entry.TheLexer)
        Val = Entry.TheLexer->isNextPPTokenLParen();
      else if (Entry.ThePTHLexer)
        Val = Entry.ThePTHLexer->isNextPPTokenLParen();
      else
        Val = Entry.TheTokenLexer->isNextTokenLParen();

      if (Val != 2)
        break;

      // Ran off the end of a source file?
      if (Entry.ThePPLexer)
        return false;
    }
  }

  // Okay, if we know that the token is a '(', lex it and return.  Otherwise we
  // have found something that isn't a '(' or we found the end of the
  // translation unit.  In either case, return false.
  return Val == 1;
}

/// HandleMacroExpandedIdentifier - If an identifier token is read that is to be
/// expanded as a macro, handle it and return the next token as 'Identifier'.
bool Preprocessor::HandleMacroExpandedIdentifier(Token &Identifier,
                                                 MacroDirective *MD) {
  MacroDirective::DefInfo Def = MD->getDefinition();
  assert(Def.isValid());
  MacroInfo *MI = Def.getMacroInfo();

  // If this is a macro expansion in the "#if !defined(x)" line for the file,
  // then the macro could expand to different things in other contexts, we need
  // to disable the optimization in this case.
  if (CurPPLexer) CurPPLexer->MIOpt.ExpandedMacro();

  // If this is a builtin macro, like __LINE__ or _Pragma, handle it specially.
  if (MI->isBuiltinMacro()) {
    if (Callbacks) Callbacks->MacroExpands(Identifier, MD,
                                           Identifier.getLocation(),/*Args=*/0);
    ExpandBuiltinMacro(Identifier);
    return false;
  }

  /// Args - If this is a function-like macro expansion, this contains,
  /// for each macro argument, the list of tokens that were provided to the
  /// invocation.
  MacroArgs *Args = 0;

  // Remember where the end of the expansion occurred.  For an object-like
  // macro, this is the identifier.  For a function-like macro, this is the ')'.
  SourceLocation ExpansionEnd = Identifier.getLocation();

  // If this is a function-like macro, read the arguments.
  if (MI->isFunctionLike()) {
    // C99 6.10.3p10: If the preprocessing token immediately after the macro
    // name isn't a '(', this macro should not be expanded.
    if (!isNextPPTokenLParen())
      return true;

    // Remember that we are now parsing the arguments to a macro invocation.
    // Preprocessor directives used inside macro arguments are not portable, and
    // this enables the warning.
    InMacroArgs = true;
    Args = ReadFunctionLikeMacroArgs(Identifier, MI, ExpansionEnd);

    // Finished parsing args.
    InMacroArgs = false;

    // If there was an error parsing the arguments, bail out.
    if (Args == 0) return false;

    ++NumFnMacroExpanded;
  } else {
    ++NumMacroExpanded;
  }

  // Notice that this macro has been used.
  markMacroAsUsed(MI);

  // Remember where the token is expanded.
  SourceLocation ExpandLoc = Identifier.getLocation();
  SourceRange ExpansionRange(ExpandLoc, ExpansionEnd);

  if (Callbacks) {
    if (InMacroArgs) {
      // We can have macro expansion inside a conditional directive while
      // reading the function macro arguments. To ensure, in that case, that
      // MacroExpands callbacks still happen in source order, queue this
      // callback to have it happen after the function macro callback.
      DelayedMacroExpandsCallbacks.push_back(
                              MacroExpandsInfo(Identifier, MD, ExpansionRange));
    } else {
      Callbacks->MacroExpands(Identifier, MD, ExpansionRange, Args);
      if (!DelayedMacroExpandsCallbacks.empty()) {
        for (unsigned i=0, e = DelayedMacroExpandsCallbacks.size(); i!=e; ++i) {
          MacroExpandsInfo &Info = DelayedMacroExpandsCallbacks[i];
          // FIXME: We lose macro args info with delayed callback.
          Callbacks->MacroExpands(Info.Tok, Info.MD, Info.Range, /*Args=*/0);
        }
        DelayedMacroExpandsCallbacks.clear();
      }
    }
  }

  // If the macro definition is ambiguous, complain.
  if (Def.getDirective()->isAmbiguous()) {
    Diag(Identifier, diag::warn_pp_ambiguous_macro)
      << Identifier.getIdentifierInfo();
    Diag(MI->getDefinitionLoc(), diag::note_pp_ambiguous_macro_chosen)
      << Identifier.getIdentifierInfo();
    for (MacroDirective::DefInfo PrevDef = Def.getPreviousDefinition();
         PrevDef && !PrevDef.isUndefined();
         PrevDef = PrevDef.getPreviousDefinition()) {
      if (PrevDef.getDirective()->isAmbiguous()) {
        Diag(PrevDef.getMacroInfo()->getDefinitionLoc(),
             diag::note_pp_ambiguous_macro_other)
          << Identifier.getIdentifierInfo();
      }
    }
  }

  // If we started lexing a macro, enter the macro expansion body.

  // If this macro expands to no tokens, don't bother to push it onto the
  // expansion stack, only to take it right back off.
  if (MI->getNumTokens() == 0) {
    // No need for arg info.
    if (Args) Args->destroy(*this);

    // Ignore this macro use, just return the next token in the current
    // buffer.
    bool HadLeadingSpace = Identifier.hasLeadingSpace();
    bool IsAtStartOfLine = Identifier.isAtStartOfLine();

    Lex(Identifier);

    // If the identifier isn't on some OTHER line, inherit the leading
    // whitespace/first-on-a-line property of this token.  This handles
    // stuff like "! XX," -> "! ," and "   XX," -> "    ,", when XX is
    // empty.
    if (!Identifier.isAtStartOfLine()) {
      if (IsAtStartOfLine) Identifier.setFlag(Token::StartOfLine);
      if (HadLeadingSpace) Identifier.setFlag(Token::LeadingSpace);
    }
    Identifier.setFlag(Token::LeadingEmptyMacro);
    ++NumFastMacroExpanded;
    return false;

  } else if (MI->getNumTokens() == 1 &&
             isTrivialSingleTokenExpansion(MI, Identifier.getIdentifierInfo(),
                                           *this)) {
    // Otherwise, if this macro expands into a single trivially-expanded
    // token: expand it now.  This handles common cases like
    // "#define VAL 42".

    // No need for arg info.
    if (Args) Args->destroy(*this);

    // Propagate the isAtStartOfLine/hasLeadingSpace markers of the macro
    // identifier to the expanded token.
    bool isAtStartOfLine = Identifier.isAtStartOfLine();
    bool hasLeadingSpace = Identifier.hasLeadingSpace();

    // Replace the result token.
    Identifier = MI->getReplacementToken(0);

    // Restore the StartOfLine/LeadingSpace markers.
    Identifier.setFlagValue(Token::StartOfLine , isAtStartOfLine);
    Identifier.setFlagValue(Token::LeadingSpace, hasLeadingSpace);

    // Update the tokens location to include both its expansion and physical
    // locations.
    SourceLocation Loc =
      SourceMgr.createExpansionLoc(Identifier.getLocation(), ExpandLoc,
                                   ExpansionEnd,Identifier.getLength());
    Identifier.setLocation(Loc);

    // If this is a disabled macro or #define X X, we must mark the result as
    // unexpandable.
    if (IdentifierInfo *NewII = Identifier.getIdentifierInfo()) {
      if (MacroInfo *NewMI = getMacroInfo(NewII))
        if (!NewMI->isEnabled() || NewMI == MI) {
          Identifier.setFlag(Token::DisableExpand);
          // Don't warn for "#define X X" like "#define bool bool" from
          // stdbool.h.
          if (NewMI != MI || MI->isFunctionLike())
            Diag(Identifier, diag::pp_disabled_macro_expansion);
        }
    }

    // Since this is not an identifier token, it can't be macro expanded, so
    // we're done.
    ++NumFastMacroExpanded;
    return false;
  }

  // Start expanding the macro.
  EnterMacro(Identifier, ExpansionEnd, MI, Args);

  // Now that the macro is at the top of the include stack, ask the
  // preprocessor to read the next token from it.
  Lex(Identifier);
  return false;
}

/// ReadFunctionLikeMacroArgs - After reading "MACRO" and knowing that the next
/// token is the '(' of the macro, this method is invoked to read all of the
/// actual arguments specified for the macro invocation.  This returns null on
/// error.
MacroArgs *Preprocessor::ReadFunctionLikeMacroArgs(Token &MacroName,
                                                   MacroInfo *MI,
                                                   SourceLocation &MacroEnd) {
  // The number of fixed arguments to parse.
  unsigned NumFixedArgsLeft = MI->getNumArgs();
  bool isVariadic = MI->isVariadic();

  // Outer loop, while there are more arguments, keep reading them.
  Token Tok;

  // Read arguments as unexpanded tokens.  This avoids issues, e.g., where
  // an argument value in a macro could expand to ',' or '(' or ')'.
  LexUnexpandedToken(Tok);
  assert(Tok.is(tok::l_paren) && "Error computing l-paren-ness?");

  // ArgTokens - Build up a list of tokens that make up each argument.  Each
  // argument is separated by an EOF token.  Use a SmallVector so we can avoid
  // heap allocations in the common case.
  SmallVector<Token, 64> ArgTokens;
  bool ContainsCodeCompletionTok = false;

  unsigned NumActuals = 0;
  while (Tok.isNot(tok::r_paren)) {
    if (ContainsCodeCompletionTok && (Tok.is(tok::eof) || Tok.is(tok::eod)))
      break;

    assert((Tok.is(tok::l_paren) || Tok.is(tok::comma)) &&
           "only expect argument separators here");

    unsigned ArgTokenStart = ArgTokens.size();
    SourceLocation ArgStartLoc = Tok.getLocation();

    // C99 6.10.3p11: Keep track of the number of l_parens we have seen.  Note
    // that we already consumed the first one.
    unsigned NumParens = 0;

    while (1) {
      // Read arguments as unexpanded tokens.  This avoids issues, e.g., where
      // an argument value in a macro could expand to ',' or '(' or ')'.
      LexUnexpandedToken(Tok);

      if (Tok.is(tok::eof) || Tok.is(tok::eod)) { // "#if f(<eof>" & "#if f(\n"
        if (!ContainsCodeCompletionTok) {
          Diag(MacroName, diag::err_unterm_macro_invoc);
          Diag(MI->getDefinitionLoc(), diag::note_macro_here)
            << MacroName.getIdentifierInfo();
          // Do not lose the EOF/EOD.  Return it to the client.
          MacroName = Tok;
          return 0;
        } else {
          // Do not lose the EOF/EOD.
          Token *Toks = new Token[1];
          Toks[0] = Tok;
          EnterTokenStream(Toks, 1, true, true);
          break;
        }
      } else if (Tok.is(tok::r_paren)) {
        // If we found the ) token, the macro arg list is done.
        if (NumParens-- == 0) {
          MacroEnd = Tok.getLocation();
          break;
        }
      } else if (Tok.is(tok::l_paren)) {
        ++NumParens;
      } else if (Tok.is(tok::comma) && NumParens == 0) {
        // Comma ends this argument if there are more fixed arguments expected.
        // However, if this is a variadic macro, and this is part of the
        // variadic part, then the comma is just an argument token.
        if (!isVariadic) break;
        if (NumFixedArgsLeft > 1)
          break;
      } else if (Tok.is(tok::comment) && !KeepMacroComments) {
        // If this is a comment token in the argument list and we're just in
        // -C mode (not -CC mode), discard the comment.
        continue;
      } else if (Tok.getIdentifierInfo() != 0) {
        // Reading macro arguments can cause macros that we are currently
        // expanding from to be popped off the expansion stack.  Doing so causes
        // them to be reenabled for expansion.  Here we record whether any
        // identifiers we lex as macro arguments correspond to disabled macros.
        // If so, we mark the token as noexpand.  This is a subtle aspect of
        // C99 6.10.3.4p2.
        if (MacroInfo *MI = getMacroInfo(Tok.getIdentifierInfo()))
          if (!MI->isEnabled())
            Tok.setFlag(Token::DisableExpand);
      } else if (Tok.is(tok::code_completion)) {
        ContainsCodeCompletionTok = true;
        if (CodeComplete)
          CodeComplete->CodeCompleteMacroArgument(MacroName.getIdentifierInfo(),
                                                  MI, NumActuals);
        // Don't mark that we reached the code-completion point because the
        // parser is going to handle the token and there will be another
        // code-completion callback.
      }

      ArgTokens.push_back(Tok);
    }

    // If this was an empty argument list foo(), don't add this as an empty
    // argument.
    if (ArgTokens.empty() && Tok.getKind() == tok::r_paren)
      break;

    // If this is not a variadic macro, and too many args were specified, emit
    // an error.
    if (!isVariadic && NumFixedArgsLeft == 0) {
      if (ArgTokens.size() != ArgTokenStart)
        ArgStartLoc = ArgTokens[ArgTokenStart].getLocation();

      if (!ContainsCodeCompletionTok) {
        // Emit the diagnostic at the macro name in case there is a missing ).
        // Emitting it at the , could be far away from the macro name.
        Diag(ArgStartLoc, diag::err_too_many_args_in_macro_invoc);
        Diag(MI->getDefinitionLoc(), diag::note_macro_here)
          << MacroName.getIdentifierInfo();
        return 0;
      }
    }

    // Empty arguments are standard in C99 and C++0x, and are supported as an extension in
    // other modes.
    if (ArgTokens.size() == ArgTokenStart && !LangOpts.C99)
      Diag(Tok, LangOpts.CPlusPlus11 ?
           diag::warn_cxx98_compat_empty_fnmacro_arg :
           diag::ext_empty_fnmacro_arg);

    // Add a marker EOF token to the end of the token list for this argument.
    Token EOFTok;
    EOFTok.startToken();
    EOFTok.setKind(tok::eof);
    EOFTok.setLocation(Tok.getLocation());
    EOFTok.setLength(0);
    ArgTokens.push_back(EOFTok);
    ++NumActuals;
    if (!ContainsCodeCompletionTok || NumFixedArgsLeft != 0) {
      assert(NumFixedArgsLeft != 0 && "Too many arguments parsed");
      --NumFixedArgsLeft;
    }
  }

  // Okay, we either found the r_paren.  Check to see if we parsed too few
  // arguments.
  unsigned MinArgsExpected = MI->getNumArgs();

  // See MacroArgs instance var for description of this.
  bool isVarargsElided = false;

  if (ContainsCodeCompletionTok) {
    // Recover from not-fully-formed macro invocation during code-completion.
    Token EOFTok;
    EOFTok.startToken();
    EOFTok.setKind(tok::eof);
    EOFTok.setLocation(Tok.getLocation());
    EOFTok.setLength(0);
    for (; NumActuals < MinArgsExpected; ++NumActuals)
      ArgTokens.push_back(EOFTok);
  }

  if (NumActuals < MinArgsExpected) {
    // There are several cases where too few arguments is ok, handle them now.
    if (NumActuals == 0 && MinArgsExpected == 1) {
      // #define A(X)  or  #define A(...)   ---> A()

      // If there is exactly one argument, and that argument is missing,
      // then we have an empty "()" argument empty list.  This is fine, even if
      // the macro expects one argument (the argument is just empty).
      isVarargsElided = MI->isVariadic();
    } else if (MI->isVariadic() &&
               (NumActuals+1 == MinArgsExpected ||  // A(x, ...) -> A(X)
                (NumActuals == 0 && MinArgsExpected == 2))) {// A(x,...) -> A()
      // Varargs where the named vararg parameter is missing: OK as extension.
      //   #define A(x, ...)
      //   A("blah")
      //
      // If the macro contains the comma pasting extension, the diagnostic
      // is suppressed; we know we'll get another diagnostic later.
      if (!MI->hasCommaPasting()) {
        Diag(Tok, diag::ext_missing_varargs_arg);
        Diag(MI->getDefinitionLoc(), diag::note_macro_here)
          << MacroName.getIdentifierInfo();
      }

      // Remember this occurred, allowing us to elide the comma when used for
      // cases like:
      //   #define A(x, foo...) blah(a, ## foo)
      //   #define B(x, ...) blah(a, ## __VA_ARGS__)
      //   #define C(...) blah(a, ## __VA_ARGS__)
      //  A(x) B(x) C()
      isVarargsElided = true;
    } else if (!ContainsCodeCompletionTok) {
      // Otherwise, emit the error.
      Diag(Tok, diag::err_too_few_args_in_macro_invoc);
      Diag(MI->getDefinitionLoc(), diag::note_macro_here)
        << MacroName.getIdentifierInfo();
      return 0;
    }

    // Add a marker EOF token to the end of the token list for this argument.
    SourceLocation EndLoc = Tok.getLocation();
    Tok.startToken();
    Tok.setKind(tok::eof);
    Tok.setLocation(EndLoc);
    Tok.setLength(0);
    ArgTokens.push_back(Tok);

    // If we expect two arguments, add both as empty.
    if (NumActuals == 0 && MinArgsExpected == 2)
      ArgTokens.push_back(Tok);

  } else if (NumActuals > MinArgsExpected && !MI->isVariadic() &&
             !ContainsCodeCompletionTok) {
    // Emit the diagnostic at the macro name in case there is a missing ).
    // Emitting it at the , could be far away from the macro name.
    Diag(MacroName, diag::err_too_many_args_in_macro_invoc);
    Diag(MI->getDefinitionLoc(), diag::note_macro_here)
      << MacroName.getIdentifierInfo();
    return 0;
  }

  return MacroArgs::create(MI, ArgTokens, isVarargsElided, *this);
}

/// \brief Keeps macro expanded tokens for TokenLexers.
//
/// Works like a stack; a TokenLexer adds the macro expanded tokens that is
/// going to lex in the cache and when it finishes the tokens are removed
/// from the end of the cache.
Token *Preprocessor::cacheMacroExpandedTokens(TokenLexer *tokLexer,
                                              ArrayRef<Token> tokens) {
  assert(tokLexer);
  if (tokens.empty())
    return 0;

  size_t newIndex = MacroExpandedTokens.size();
  bool cacheNeedsToGrow = tokens.size() >
                      MacroExpandedTokens.capacity()-MacroExpandedTokens.size(); 
  MacroExpandedTokens.append(tokens.begin(), tokens.end());

  if (cacheNeedsToGrow) {
    // Go through all the TokenLexers whose 'Tokens' pointer points in the
    // buffer and update the pointers to the (potential) new buffer array.
    for (unsigned i = 0, e = MacroExpandingLexersStack.size(); i != e; ++i) {
      TokenLexer *prevLexer;
      size_t tokIndex;
      llvm::tie(prevLexer, tokIndex) = MacroExpandingLexersStack[i];
      prevLexer->Tokens = MacroExpandedTokens.data() + tokIndex;
    }
  }

  MacroExpandingLexersStack.push_back(std::make_pair(tokLexer, newIndex));
  return MacroExpandedTokens.data() + newIndex;
}

void Preprocessor::removeCachedMacroExpandedTokensOfLastLexer() {
  assert(!MacroExpandingLexersStack.empty());
  size_t tokIndex = MacroExpandingLexersStack.back().second;
  assert(tokIndex < MacroExpandedTokens.size());
  // Pop the cached macro expanded tokens from the end.
  MacroExpandedTokens.resize(tokIndex);
  MacroExpandingLexersStack.pop_back();
}

/// ComputeDATE_TIME - Compute the current time, enter it into the specified
/// scratch buffer, then return DATELoc/TIMELoc locations with the position of
/// the identifier tokens inserted.
static void ComputeDATE_TIME(SourceLocation &DATELoc, SourceLocation &TIMELoc,
                             Preprocessor &PP) {
  time_t TT = time(0);
  struct tm *TM = localtime(&TT);

  static const char * const Months[] = {
    "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"
  };

  {
    SmallString<32> TmpBuffer;
    llvm::raw_svector_ostream TmpStream(TmpBuffer);
    TmpStream << llvm::format("\"%s %2d %4d\"", Months[TM->tm_mon],
                              TM->tm_mday, TM->tm_year + 1900);
    Token TmpTok;
    TmpTok.startToken();
    PP.CreateString(TmpStream.str(), TmpTok);
    DATELoc = TmpTok.getLocation();
  }

  {
    SmallString<32> TmpBuffer;
    llvm::raw_svector_ostream TmpStream(TmpBuffer);
    TmpStream << llvm::format("\"%02d:%02d:%02d\"",
                              TM->tm_hour, TM->tm_min, TM->tm_sec);
    Token TmpTok;
    TmpTok.startToken();
    PP.CreateString(TmpStream.str(), TmpTok);
    TIMELoc = TmpTok.getLocation();
  }
}


/// HasFeature - Return true if we recognize and implement the feature
/// specified by the identifier as a standard language feature.
static bool HasFeature(const Preprocessor &PP, const IdentifierInfo *II) {
  const LangOptions &LangOpts = PP.getLangOpts();
  StringRef Feature = II->getName();

  // Normalize the feature name, __foo__ becomes foo.
  if (Feature.startswith("__") && Feature.endswith("__") && Feature.size() >= 4)
    Feature = Feature.substr(2, Feature.size() - 4);

  return llvm::StringSwitch<bool>(Feature)
           .Case("address_sanitizer", LangOpts.Sanitize.Address)
           .Case("attribute_analyzer_noreturn", true)
           .Case("attribute_availability", true)
           .Case("attribute_availability_with_message", true)
           .Case("attribute_cf_returns_not_retained", true)
           .Case("attribute_cf_returns_retained", true)
           .Case("attribute_deprecated_with_message", true)
           .Case("attribute_ext_vector_type", true)
           .Case("attribute_ns_returns_not_retained", true)
           .Case("attribute_ns_returns_retained", true)
           .Case("attribute_ns_consumes_self", true)
           .Case("attribute_ns_consumed", true)
           .Case("attribute_cf_consumed", true)
           .Case("attribute_objc_ivar_unused", true)
           .Case("attribute_objc_method_family", true)
           .Case("attribute_overloadable", true)
           .Case("attribute_unavailable_with_message", true)
           .Case("attribute_unused_on_fields", true)
           .Case("blocks", LangOpts.Blocks)
           .Case("cxx_exceptions", LangOpts.Exceptions)
           .Case("cxx_rtti", LangOpts.RTTI)
           .Case("enumerator_attributes", true)
           .Case("memory_sanitizer", LangOpts.Sanitize.Memory)
           .Case("thread_sanitizer", LangOpts.Sanitize.Thread)
           // Objective-C features
           .Case("objc_arr", LangOpts.ObjCAutoRefCount) // FIXME: REMOVE?
           .Case("objc_arc", LangOpts.ObjCAutoRefCount)
           .Case("objc_arc_weak", LangOpts.ObjCARCWeak)
           .Case("objc_default_synthesize_properties", LangOpts.ObjC2)
           .Case("objc_fixed_enum", LangOpts.ObjC2)
           .Case("objc_instancetype", LangOpts.ObjC2)
           .Case("objc_modules", LangOpts.ObjC2 && LangOpts.Modules)
           .Case("objc_nonfragile_abi", LangOpts.ObjCRuntime.isNonFragile())
           .Case("objc_property_explicit_atomic", true) // Does clang support explicit "atomic" keyword?
           .Case("objc_weak_class", LangOpts.ObjCRuntime.hasWeakClassImport())
           .Case("ownership_holds", true)
           .Case("ownership_returns", true)
           .Case("ownership_takes", true)
           .Case("objc_bool", true)
           .Case("objc_subscripting", LangOpts.ObjCRuntime.isNonFragile())
           .Case("objc_array_literals", LangOpts.ObjC2)
           .Case("objc_dictionary_literals", LangOpts.ObjC2)
           .Case("objc_boxed_expressions", LangOpts.ObjC2)
           .Case("arc_cf_code_audited", true)
           // C11 features
           .Case("c_alignas", LangOpts.C11)
           .Case("c_atomic", LangOpts.C11)
           .Case("c_generic_selections", LangOpts.C11)
           .Case("c_static_assert", LangOpts.C11)
           .Case("c_thread_local", 
                 LangOpts.C11 && PP.getTargetInfo().isTLSSupported())
           // C++11 features
           .Case("cxx_access_control_sfinae", LangOpts.CPlusPlus11)
           .Case("cxx_alias_templates", LangOpts.CPlusPlus11)
           .Case("cxx_alignas", LangOpts.CPlusPlus11)
           .Case("cxx_atomic", LangOpts.CPlusPlus11)
           .Case("cxx_attributes", LangOpts.CPlusPlus11)
           .Case("cxx_auto_type", LangOpts.CPlusPlus11)
           .Case("cxx_constexpr", LangOpts.CPlusPlus11)
           .Case("cxx_decltype", LangOpts.CPlusPlus11)
           .Case("cxx_decltype_incomplete_return_types", LangOpts.CPlusPlus11)
           .Case("cxx_default_function_template_args", LangOpts.CPlusPlus11)
           .Case("cxx_defaulted_functions", LangOpts.CPlusPlus11)
           .Case("cxx_delegating_constructors", LangOpts.CPlusPlus11)
           .Case("cxx_deleted_functions", LangOpts.CPlusPlus11)
           .Case("cxx_explicit_conversions", LangOpts.CPlusPlus11)
           .Case("cxx_generalized_initializers", LangOpts.CPlusPlus11)
           .Case("cxx_implicit_moves", LangOpts.CPlusPlus11)
           .Case("cxx_inheriting_constructors", LangOpts.CPlusPlus11)
           .Case("cxx_inline_namespaces", LangOpts.CPlusPlus11)
           .Case("cxx_lambdas", LangOpts.CPlusPlus11)
           .Case("cxx_local_type_template_args", LangOpts.CPlusPlus11)
           .Case("cxx_nonstatic_member_init", LangOpts.CPlusPlus11)
           .Case("cxx_noexcept", LangOpts.CPlusPlus11)
           .Case("cxx_nullptr", LangOpts.CPlusPlus11)
           .Case("cxx_override_control", LangOpts.CPlusPlus11)
           .Case("cxx_range_for", LangOpts.CPlusPlus11)
           .Case("cxx_raw_string_literals", LangOpts.CPlusPlus11)
           .Case("cxx_reference_qualified_functions", LangOpts.CPlusPlus11)
           .Case("cxx_rvalue_references", LangOpts.CPlusPlus11)
           .Case("cxx_strong_enums", LangOpts.CPlusPlus11)
           .Case("cxx_static_assert", LangOpts.CPlusPlus11)
           .Case("cxx_thread_local", 
                 LangOpts.CPlusPlus11 && PP.getTargetInfo().isTLSSupported())
           .Case("cxx_trailing_return", LangOpts.CPlusPlus11)
           .Case("cxx_unicode_literals", LangOpts.CPlusPlus11)
           .Case("cxx_unrestricted_unions", LangOpts.CPlusPlus11)
           .Case("cxx_user_literals", LangOpts.CPlusPlus11)
           .Case("cxx_variadic_templates", LangOpts.CPlusPlus11)
           // C++1y features
           .Case("cxx_binary_literals", LangOpts.CPlusPlus1y)
           //.Case("cxx_contextual_conversions", LangOpts.CPlusPlus1y)
           //.Case("cxx_generalized_capture", LangOpts.CPlusPlus1y)
           //.Case("cxx_generic_lambda", LangOpts.CPlusPlus1y)
           //.Case("cxx_relaxed_constexpr", LangOpts.CPlusPlus1y)
           //.Case("cxx_return_type_deduction", LangOpts.CPlusPlus1y)
           //.Case("cxx_runtime_array", LangOpts.CPlusPlus1y)
           .Case("cxx_aggregate_nsdmi", LangOpts.CPlusPlus1y)
           //.Case("cxx_variable_templates", LangOpts.CPlusPlus1y)
           // Type traits
           .Case("has_nothrow_assign", LangOpts.CPlusPlus)
           .Case("has_nothrow_copy", LangOpts.CPlusPlus)
           .Case("has_nothrow_constructor", LangOpts.CPlusPlus)
           .Case("has_trivial_assign", LangOpts.CPlusPlus)
           .Case("has_trivial_copy", LangOpts.CPlusPlus)
           .Case("has_trivial_constructor", LangOpts.CPlusPlus)
           .Case("has_trivial_destructor", LangOpts.CPlusPlus)
           .Case("has_virtual_destructor", LangOpts.CPlusPlus)
           .Case("is_abstract", LangOpts.CPlusPlus)
           .Case("is_base_of", LangOpts.CPlusPlus)
           .Case("is_class", LangOpts.CPlusPlus)
           .Case("is_convertible_to", LangOpts.CPlusPlus)
           .Case("is_empty", LangOpts.CPlusPlus)
           .Case("is_enum", LangOpts.CPlusPlus)
           .Case("is_final", LangOpts.CPlusPlus)
           .Case("is_literal", LangOpts.CPlusPlus)
           .Case("is_standard_layout", LangOpts.CPlusPlus)
           .Case("is_pod", LangOpts.CPlusPlus)
           .Case("is_polymorphic", LangOpts.CPlusPlus)
           .Case("is_trivial", LangOpts.CPlusPlus)
           .Case("is_trivially_assignable", LangOpts.CPlusPlus)
           .Case("is_trivially_constructible", LangOpts.CPlusPlus)
           .Case("is_trivially_copyable", LangOpts.CPlusPlus)
           .Case("is_union", LangOpts.CPlusPlus)
           .Case("modules", LangOpts.Modules)
           .Case("tls", PP.getTargetInfo().isTLSSupported())
           .Case("underlying_type", LangOpts.CPlusPlus)
           .Default(false);
}

/// HasExtension - Return true if we recognize and implement the feature
/// specified by the identifier, either as an extension or a standard language
/// feature.
static bool HasExtension(const Preprocessor &PP, const IdentifierInfo *II) {
  if (HasFeature(PP, II))
    return true;

  // If the use of an extension results in an error diagnostic, extensions are
  // effectively unavailable, so just return false here.
  if (PP.getDiagnostics().getExtensionHandlingBehavior() ==
      DiagnosticsEngine::Ext_Error)
    return false;

  const LangOptions &LangOpts = PP.getLangOpts();
  StringRef Extension = II->getName();

  // Normalize the extension name, __foo__ becomes foo.
  if (Extension.startswith("__") && Extension.endswith("__") &&
      Extension.size() >= 4)
    Extension = Extension.substr(2, Extension.size() - 4);

  // Because we inherit the feature list from HasFeature, this string switch
  // must be less restrictive than HasFeature's.
  return llvm::StringSwitch<bool>(Extension)
           // C11 features supported by other languages as extensions.
           .Case("c_alignas", true)
           .Case("c_atomic", true)
           .Case("c_generic_selections", true)
           .Case("c_static_assert", true)
           // C++11 features supported by other languages as extensions.
           .Case("cxx_atomic", LangOpts.CPlusPlus)
           .Case("cxx_deleted_functions", LangOpts.CPlusPlus)
           .Case("cxx_explicit_conversions", LangOpts.CPlusPlus)
           .Case("cxx_inline_namespaces", LangOpts.CPlusPlus)
           .Case("cxx_local_type_template_args", LangOpts.CPlusPlus)
           .Case("cxx_nonstatic_member_init", LangOpts.CPlusPlus)
           .Case("cxx_override_control", LangOpts.CPlusPlus)
           .Case("cxx_range_for", LangOpts.CPlusPlus)
           .Case("cxx_reference_qualified_functions", LangOpts.CPlusPlus)
           .Case("cxx_rvalue_references", LangOpts.CPlusPlus)
           // C++1y features supported by other languages as extensions.
           .Case("cxx_binary_literals", true)
           .Default(false);
}

/// HasAttribute -  Return true if we recognize and implement the attribute
/// specified by the given identifier.
static bool HasAttribute(const IdentifierInfo *II) {
  StringRef Name = II->getName();
  // Normalize the attribute name, __foo__ becomes foo.
  if (Name.startswith("__") && Name.endswith("__") && Name.size() >= 4)
    Name = Name.substr(2, Name.size() - 4);

  // FIXME: Do we need to handle namespaces here?
  return llvm::StringSwitch<bool>(Name)
#include "clang/Lex/AttrSpellings.inc"
        .Default(false);
}

/// EvaluateHasIncludeCommon - Process a '__has_include("path")'
/// or '__has_include_next("path")' expression.
/// Returns true if successful.
static bool EvaluateHasIncludeCommon(Token &Tok,
                                     IdentifierInfo *II, Preprocessor &PP,
                                     const DirectoryLookup *LookupFrom) {
  // Save the location of the current token.  If a '(' is later found, use
  // that location.  If not, use the end of this location instead.
  SourceLocation LParenLoc = Tok.getLocation();

  // These expressions are only allowed within a preprocessor directive.
  if (!PP.isParsingIfOrElifDirective()) {
    PP.Diag(LParenLoc, diag::err_pp_directive_required) << II->getName();
    return false;
  }

  // Get '('.
  PP.LexNonComment(Tok);

  // Ensure we have a '('.
  if (Tok.isNot(tok::l_paren)) {
    // No '(', use end of last token.
    LParenLoc = PP.getLocForEndOfToken(LParenLoc);
    PP.Diag(LParenLoc, diag::err_pp_missing_lparen) << II->getName();
    // If the next token looks like a filename or the start of one,
    // assume it is and process it as such.
    if (!Tok.is(tok::angle_string_literal) && !Tok.is(tok::string_literal) &&
        !Tok.is(tok::less))
      return false;
  } else {
    // Save '(' location for possible missing ')' message.
    LParenLoc = Tok.getLocation();

    if (PP.getCurrentLexer()) {
      // Get the file name.
      PP.getCurrentLexer()->LexIncludeFilename(Tok);
    } else {
      // We're in a macro, so we can't use LexIncludeFilename; just
      // grab the next token.
      PP.Lex(Tok);
    }
  }

  // Reserve a buffer to get the spelling.
  SmallString<128> FilenameBuffer;
  StringRef Filename;
  SourceLocation EndLoc;
  
  switch (Tok.getKind()) {
  case tok::eod:
    // If the token kind is EOD, the error has already been diagnosed.
    return false;

  case tok::angle_string_literal:
  case tok::string_literal: {
    bool Invalid = false;
    Filename = PP.getSpelling(Tok, FilenameBuffer, &Invalid);
    if (Invalid)
      return false;
    break;
  }

  case tok::less:
    // This could be a <foo/bar.h> file coming from a macro expansion.  In this
    // case, glue the tokens together into FilenameBuffer and interpret those.
    FilenameBuffer.push_back('<');
    if (PP.ConcatenateIncludeName(FilenameBuffer, EndLoc)) {
      // Let the caller know a <eod> was found by changing the Token kind.
      Tok.setKind(tok::eod);
      return false;   // Found <eod> but no ">"?  Diagnostic already emitted.
    }
    Filename = FilenameBuffer.str();
    break;
  default:
    PP.Diag(Tok.getLocation(), diag::err_pp_expects_filename);
    return false;
  }

  SourceLocation FilenameLoc = Tok.getLocation();

  // Get ')'.
  PP.LexNonComment(Tok);

  // Ensure we have a trailing ).
  if (Tok.isNot(tok::r_paren)) {
    PP.Diag(PP.getLocForEndOfToken(FilenameLoc), diag::err_pp_missing_rparen)
        << II->getName();
    PP.Diag(LParenLoc, diag::note_matching) << "(";
    return false;
  }

  bool isAngled = PP.GetIncludeFilenameSpelling(Tok.getLocation(), Filename);
  // If GetIncludeFilenameSpelling set the start ptr to null, there was an
  // error.
  if (Filename.empty())
    return false;

  // Search include directories.
  const DirectoryLookup *CurDir;
  const FileEntry *File =
      PP.LookupFile(Filename, isAngled, LookupFrom, CurDir, NULL, NULL, NULL);

  // Get the result value.  A result of true means the file exists.
  return File != 0;
}

/// EvaluateHasInclude - Process a '__has_include("path")' expression.
/// Returns true if successful.
static bool EvaluateHasInclude(Token &Tok, IdentifierInfo *II,
                               Preprocessor &PP) {
  return EvaluateHasIncludeCommon(Tok, II, PP, NULL);
}

/// EvaluateHasIncludeNext - Process '__has_include_next("path")' expression.
/// Returns true if successful.
static bool EvaluateHasIncludeNext(Token &Tok,
                                   IdentifierInfo *II, Preprocessor &PP) {
  // __has_include_next is like __has_include, except that we start
  // searching after the current found directory.  If we can't do this,
  // issue a diagnostic.
  const DirectoryLookup *Lookup = PP.GetCurDirLookup();
  if (PP.isInPrimaryFile()) {
    Lookup = 0;
    PP.Diag(Tok, diag::pp_include_next_in_primary);
  } else if (Lookup == 0) {
    PP.Diag(Tok, diag::pp_include_next_absolute_path);
  } else {
    // Start looking up in the next directory.
    ++Lookup;
  }

  return EvaluateHasIncludeCommon(Tok, II, PP, Lookup);
}

/// \brief Process __building_module(identifier) expression.
/// \returns true if we are building the named module, false otherwise.
static bool EvaluateBuildingModule(Token &Tok,
                                   IdentifierInfo *II, Preprocessor &PP) {
  // Get '('.
  PP.LexNonComment(Tok);

  // Ensure we have a '('.
  if (Tok.isNot(tok::l_paren)) {
    PP.Diag(Tok.getLocation(), diag::err_pp_missing_lparen) << II->getName();
    return false;
  }

  // Save '(' location for possible missing ')' message.
  SourceLocation LParenLoc = Tok.getLocation();

  // Get the module name.
  PP.LexNonComment(Tok);

  // Ensure that we have an identifier.
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::err_expected_id_building_module);
    return false;
  }

  bool Result
    = Tok.getIdentifierInfo()->getName() == PP.getLangOpts().CurrentModule;

  // Get ')'.
  PP.LexNonComment(Tok);

  // Ensure we have a trailing ).
  if (Tok.isNot(tok::r_paren)) {
    PP.Diag(Tok.getLocation(), diag::err_pp_missing_rparen) << II->getName();
    PP.Diag(LParenLoc, diag::note_matching) << "(";
    return false;
  }

  return Result;
}

/// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
/// as a builtin macro, handle it and return the next token as 'Tok'.
void Preprocessor::ExpandBuiltinMacro(Token &Tok) {
  // Figure out which token this is.
  IdentifierInfo *II = Tok.getIdentifierInfo();
  assert(II && "Can't be a macro without id info!");

  // If this is an _Pragma or Microsoft __pragma directive, expand it,
  // invoke the pragma handler, then lex the token after it.
  if (II == Ident_Pragma)
    return Handle_Pragma(Tok);
  else if (II == Ident__pragma) // in non-MS mode this is null
    return HandleMicrosoft__pragma(Tok);

  ++NumBuiltinMacroExpanded;

  SmallString<128> TmpBuffer;
  llvm::raw_svector_ostream OS(TmpBuffer);

  // Set up the return result.
  Tok.setIdentifierInfo(0);
  Tok.clearFlag(Token::NeedsCleaning);

  if (II == Ident__LINE__) {
    // C99 6.10.8: "__LINE__: The presumed line number (within the current
    // source file) of the current source line (an integer constant)".  This can
    // be affected by #line.
    SourceLocation Loc = Tok.getLocation();

    // Advance to the location of the first _, this might not be the first byte
    // of the token if it starts with an escaped newline.
    Loc = AdvanceToTokenCharacter(Loc, 0);

    // One wrinkle here is that GCC expands __LINE__ to location of the *end* of
    // a macro expansion.  This doesn't matter for object-like macros, but
    // can matter for a function-like macro that expands to contain __LINE__.
    // Skip down through expansion points until we find a file loc for the
    // end of the expansion history.
    Loc = SourceMgr.getExpansionRange(Loc).second;
    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Loc);

    // __LINE__ expands to a simple numeric value.
    OS << (PLoc.isValid()? PLoc.getLine() : 1);
    Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__FILE__ || II == Ident__BASE_FILE__) {
    // C99 6.10.8: "__FILE__: The presumed name of the current source file (a
    // character string literal)". This can be affected by #line.
    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Tok.getLocation());

    // __BASE_FILE__ is a GNU extension that returns the top of the presumed
    // #include stack instead of the current file.
    if (II == Ident__BASE_FILE__ && PLoc.isValid()) {
      SourceLocation NextLoc = PLoc.getIncludeLoc();
      while (NextLoc.isValid()) {
        PLoc = SourceMgr.getPresumedLoc(NextLoc);
        if (PLoc.isInvalid())
          break;
        
        NextLoc = PLoc.getIncludeLoc();
      }
    }

    // Escape this filename.  Turn '\' -> '\\' '"' -> '\"'
    SmallString<128> FN;
    if (PLoc.isValid()) {
      FN += PLoc.getFilename();
      Lexer::Stringify(FN);
      OS << '"' << FN.str() << '"';
    }
    Tok.setKind(tok::string_literal);
  } else if (II == Ident__DATE__) {
    if (!DATELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"Mmm dd yyyy\""));
    Tok.setLocation(SourceMgr.createExpansionLoc(DATELoc, Tok.getLocation(),
                                                 Tok.getLocation(),
                                                 Tok.getLength()));
    return;
  } else if (II == Ident__TIME__) {
    if (!TIMELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"hh:mm:ss\""));
    Tok.setLocation(SourceMgr.createExpansionLoc(TIMELoc, Tok.getLocation(),
                                                 Tok.getLocation(),
                                                 Tok.getLength()));
    return;
  } else if (II == Ident__INCLUDE_LEVEL__) {
    // Compute the presumed include depth of this token.  This can be affected
    // by GNU line markers.
    unsigned Depth = 0;

    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Tok.getLocation());
    if (PLoc.isValid()) {
      PLoc = SourceMgr.getPresumedLoc(PLoc.getIncludeLoc());
      for (; PLoc.isValid(); ++Depth)
        PLoc = SourceMgr.getPresumedLoc(PLoc.getIncludeLoc());
    }

    // __INCLUDE_LEVEL__ expands to a simple numeric value.
    OS << Depth;
    Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__TIMESTAMP__) {
    // MSVC, ICC, GCC, VisualAge C++ extension.  The generated string should be
    // of the form "Ddd Mmm dd hh::mm::ss yyyy", which is returned by asctime.

    // Get the file that we are lexing out of.  If we're currently lexing from
    // a macro, dig into the include stack.
    const FileEntry *CurFile = 0;
    PreprocessorLexer *TheLexer = getCurrentFileLexer();

    if (TheLexer)
      CurFile = SourceMgr.getFileEntryForID(TheLexer->getFileID());

    const char *Result;
    if (CurFile) {
      time_t TT = CurFile->getModificationTime();
      struct tm *TM = localtime(&TT);
      Result = asctime(TM);
    } else {
      Result = "??? ??? ?? ??:??:?? ????\n";
    }
    // Surround the string with " and strip the trailing newline.
    OS << '"' << StringRef(Result, strlen(Result)-1) << '"';
    Tok.setKind(tok::string_literal);
  } else if (II == Ident__COUNTER__) {
    // __COUNTER__ expands to a simple numeric value.
    OS << CounterValue++;
    Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__has_feature   ||
             II == Ident__has_extension ||
             II == Ident__has_builtin   ||
             II == Ident__has_attribute) {
    // The argument to these builtins should be a parenthesized identifier.
    SourceLocation StartLoc = Tok.getLocation();

    bool IsValid = false;
    IdentifierInfo *FeatureII = 0;

    // Read the '('.
    LexUnexpandedToken(Tok);
    if (Tok.is(tok::l_paren)) {
      // Read the identifier
      LexUnexpandedToken(Tok);
      if (Tok.is(tok::identifier) || Tok.is(tok::kw_const)) {
        FeatureII = Tok.getIdentifierInfo();

        // Read the ')'.
        LexUnexpandedToken(Tok);
        if (Tok.is(tok::r_paren))
          IsValid = true;
      }
    }

    bool Value = false;
    if (!IsValid)
      Diag(StartLoc, diag::err_feature_check_malformed);
    else if (II == Ident__has_builtin) {
      // Check for a builtin is trivial.
      Value = FeatureII->getBuiltinID() != 0;
    } else if (II == Ident__has_attribute)
      Value = HasAttribute(FeatureII);
    else if (II == Ident__has_extension)
      Value = HasExtension(*this, FeatureII);
    else {
      assert(II == Ident__has_feature && "Must be feature check");
      Value = HasFeature(*this, FeatureII);
    }

    OS << (int)Value;
    if (IsValid)
      Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__has_include ||
             II == Ident__has_include_next) {
    // The argument to these two builtins should be a parenthesized
    // file name string literal using angle brackets (<>) or
    // double-quotes ("").
    bool Value;
    if (II == Ident__has_include)
      Value = EvaluateHasInclude(Tok, II, *this);
    else
      Value = EvaluateHasIncludeNext(Tok, II, *this);
    OS << (int)Value;
    if (Tok.is(tok::r_paren))
      Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__has_warning) {
    // The argument should be a parenthesized string literal.
    // The argument to these builtins should be a parenthesized identifier.
    SourceLocation StartLoc = Tok.getLocation();    
    bool IsValid = false;
    bool Value = false;
    // Read the '('.
    LexUnexpandedToken(Tok);
    do {
      if (Tok.isNot(tok::l_paren)) {
        Diag(StartLoc, diag::err_warning_check_malformed);
        break;
      }

      LexUnexpandedToken(Tok);
      std::string WarningName;
      SourceLocation StrStartLoc = Tok.getLocation();
      if (!FinishLexStringLiteral(Tok, WarningName, "'__has_warning'",
                                  /*MacroExpansion=*/false)) {
        // Eat tokens until ')'.
        while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::eod) &&
               Tok.isNot(tok::eof))
          LexUnexpandedToken(Tok);
        break;
      }

      // Is the end a ')'?
      if (!(IsValid = Tok.is(tok::r_paren))) {
        Diag(StartLoc, diag::err_warning_check_malformed);
        break;
      }

      if (WarningName.size() < 3 || WarningName[0] != '-' ||
          WarningName[1] != 'W') {
        Diag(StrStartLoc, diag::warn_has_warning_invalid_option);
        break;
      }

      // Finally, check if the warning flags maps to a diagnostic group.
      // We construct a SmallVector here to talk to getDiagnosticIDs().
      // Although we don't use the result, this isn't a hot path, and not
      // worth special casing.
      SmallVector<diag::kind, 10> Diags;
      Value = !getDiagnostics().getDiagnosticIDs()->
        getDiagnosticsInGroup(WarningName.substr(2), Diags);
    } while (false);

    OS << (int)Value;
    if (IsValid)
      Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__building_module) {
    // The argument to this builtin should be an identifier. The
    // builtin evaluates to 1 when that identifier names the module we are
    // currently building.
    OS << (int)EvaluateBuildingModule(Tok, II, *this);
    Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__MODULE__) {
    // The current module as an identifier.
    OS << getLangOpts().CurrentModule;
    IdentifierInfo *ModuleII = getIdentifierInfo(getLangOpts().CurrentModule);
    Tok.setIdentifierInfo(ModuleII);
    Tok.setKind(ModuleII->getTokenID());
  } else {
    llvm_unreachable("Unknown identifier!");
  }
  CreateString(OS.str(), Tok, Tok.getLocation(), Tok.getLocation());
}

void Preprocessor::markMacroAsUsed(MacroInfo *MI) {
  // If the 'used' status changed, and the macro requires 'unused' warning,
  // remove its SourceLocation from the warn-for-unused-macro locations.
  if (MI->isWarnIfUnused() && !MI->isUsed())
    WarnUnusedMacroLocs.erase(MI->getDefinitionLoc());
  MI->setIsUsed(true);
}
