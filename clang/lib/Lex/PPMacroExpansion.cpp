//===--- MacroExpansion.cpp - Top level Macro Expansion -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the top level handling of macro expansion for the
// preprocessor.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/MacroArgs.h"
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

  // Microsoft Extensions.
  if (LangOpts.MicrosoftExt) {
    Ident__identifier = RegisterBuiltinMacro(*this, "__identifier");
    Ident__pragma = RegisterBuiltinMacro(*this, "__pragma");
  } else {
    Ident__identifier = nullptr;
    Ident__pragma = nullptr;
  }

  // Clang Extensions.
  Ident__has_feature      = RegisterBuiltinMacro(*this, "__has_feature");
  Ident__has_extension    = RegisterBuiltinMacro(*this, "__has_extension");
  Ident__has_builtin      = RegisterBuiltinMacro(*this, "__has_builtin");
  Ident__has_attribute    = RegisterBuiltinMacro(*this, "__has_attribute");
  Ident__has_include      = RegisterBuiltinMacro(*this, "__has_include");
  Ident__has_include_next = RegisterBuiltinMacro(*this, "__has_include_next");
  Ident__has_warning      = RegisterBuiltinMacro(*this, "__has_warning");
  Ident__is_identifier    = RegisterBuiltinMacro(*this, "__is_identifier");

  // Modules.
  if (LangOpts.Modules) {
    Ident__building_module  = RegisterBuiltinMacro(*this, "__building_module");

    // __MODULE__
    if (!LangOpts.CurrentModule.empty())
      Ident__MODULE__ = RegisterBuiltinMacro(*this, "__MODULE__");
    else
      Ident__MODULE__ = nullptr;
  } else {
    Ident__building_module = nullptr;
    Ident__MODULE__ = nullptr;
  }
}

/// isTrivialSingleTokenExpansion - Return true if MI, which has a single token
/// in its expansion, currently expands to that token literally.
static bool isTrivialSingleTokenExpansion(const MacroInfo *MI,
                                          const IdentifierInfo *MacroIdent,
                                          Preprocessor &PP) {
  IdentifierInfo *II = MI->getReplacementToken(0).getIdentifierInfo();

  // If the token isn't an identifier, it's always literally expanded.
  if (!II) return true;

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
                                           Identifier.getLocation(),
                                           /*Args=*/nullptr);
    ExpandBuiltinMacro(Identifier);
    return true;
  }

  /// Args - If this is a function-like macro expansion, this contains,
  /// for each macro argument, the list of tokens that were provided to the
  /// invocation.
  MacroArgs *Args = nullptr;

  // Remember where the end of the expansion occurred.  For an object-like
  // macro, this is the identifier.  For a function-like macro, this is the ')'.
  SourceLocation ExpansionEnd = Identifier.getLocation();

  // If this is a function-like macro, read the arguments.
  if (MI->isFunctionLike()) {
    // Remember that we are now parsing the arguments to a macro invocation.
    // Preprocessor directives used inside macro arguments are not portable, and
    // this enables the warning.
    InMacroArgs = true;
    Args = ReadFunctionLikeMacroArgs(Identifier, MI, ExpansionEnd);

    // Finished parsing args.
    InMacroArgs = false;

    // If there was an error parsing the arguments, bail out.
    if (!Args) return true;

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
          Callbacks->MacroExpands(Info.Tok, Info.MD, Info.Range,
                                  /*Args=*/nullptr);
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
      Diag(PrevDef.getMacroInfo()->getDefinitionLoc(),
           diag::note_pp_ambiguous_macro_other)
        << Identifier.getIdentifierInfo();
      if (!PrevDef.getDirective()->isAmbiguous())
        break;
    }
  }

  // If we started lexing a macro, enter the macro expansion body.

  // If this macro expands to no tokens, don't bother to push it onto the
  // expansion stack, only to take it right back off.
  if (MI->getNumTokens() == 0) {
    // No need for arg info.
    if (Args) Args->destroy(*this);

    // Propagate whitespace info as if we had pushed, then popped,
    // a macro context.
    Identifier.setFlag(Token::LeadingEmptyMacro);
    PropagateLineStartLeadingSpaceInfo(Identifier);
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
    return true;
  }

  // Start expanding the macro.
  EnterMacro(Identifier, ExpansionEnd, MI, Args);
  return false;
}

enum Bracket {
  Brace,
  Paren
};

/// CheckMatchedBrackets - Returns true if the braces and parentheses in the
/// token vector are properly nested.
static bool CheckMatchedBrackets(const SmallVectorImpl<Token> &Tokens) {
  SmallVector<Bracket, 8> Brackets;
  for (SmallVectorImpl<Token>::const_iterator I = Tokens.begin(),
                                              E = Tokens.end();
       I != E; ++I) {
    if (I->is(tok::l_paren)) {
      Brackets.push_back(Paren);
    } else if (I->is(tok::r_paren)) {
      if (Brackets.empty() || Brackets.back() == Brace)
        return false;
      Brackets.pop_back();
    } else if (I->is(tok::l_brace)) {
      Brackets.push_back(Brace);
    } else if (I->is(tok::r_brace)) {
      if (Brackets.empty() || Brackets.back() == Paren)
        return false;
      Brackets.pop_back();
    }
  }
  if (!Brackets.empty())
    return false;
  return true;
}

/// GenerateNewArgTokens - Returns true if OldTokens can be converted to a new
/// vector of tokens in NewTokens.  The new number of arguments will be placed
/// in NumArgs and the ranges which need to surrounded in parentheses will be
/// in ParenHints.
/// Returns false if the token stream cannot be changed.  If this is because
/// of an initializer list starting a macro argument, the range of those
/// initializer lists will be place in InitLists.
static bool GenerateNewArgTokens(Preprocessor &PP,
                                 SmallVectorImpl<Token> &OldTokens,
                                 SmallVectorImpl<Token> &NewTokens,
                                 unsigned &NumArgs,
                                 SmallVectorImpl<SourceRange> &ParenHints,
                                 SmallVectorImpl<SourceRange> &InitLists) {
  if (!CheckMatchedBrackets(OldTokens))
    return false;

  // Once it is known that the brackets are matched, only a simple count of the
  // braces is needed.
  unsigned Braces = 0;

  // First token of a new macro argument.
  SmallVectorImpl<Token>::iterator ArgStartIterator = OldTokens.begin();

  // First closing brace in a new macro argument.  Used to generate
  // SourceRanges for InitLists.
  SmallVectorImpl<Token>::iterator ClosingBrace = OldTokens.end();
  NumArgs = 0;
  Token TempToken;
  // Set to true when a macro separator token is found inside a braced list.
  // If true, the fixed argument spans multiple old arguments and ParenHints
  // will be updated.
  bool FoundSeparatorToken = false;
  for (SmallVectorImpl<Token>::iterator I = OldTokens.begin(),
                                        E = OldTokens.end();
       I != E; ++I) {
    if (I->is(tok::l_brace)) {
      ++Braces;
    } else if (I->is(tok::r_brace)) {
      --Braces;
      if (Braces == 0 && ClosingBrace == E && FoundSeparatorToken)
        ClosingBrace = I;
    } else if (I->is(tok::eof)) {
      // EOF token is used to separate macro arguments
      if (Braces != 0) {
        // Assume comma separator is actually braced list separator and change
        // it back to a comma.
        FoundSeparatorToken = true;
        I->setKind(tok::comma);
        I->setLength(1);
      } else { // Braces == 0
        // Separator token still separates arguments.
        ++NumArgs;

        // If the argument starts with a brace, it can't be fixed with
        // parentheses.  A different diagnostic will be given.
        if (FoundSeparatorToken && ArgStartIterator->is(tok::l_brace)) {
          InitLists.push_back(
              SourceRange(ArgStartIterator->getLocation(),
                          PP.getLocForEndOfToken(ClosingBrace->getLocation())));
          ClosingBrace = E;
        }

        // Add left paren
        if (FoundSeparatorToken) {
          TempToken.startToken();
          TempToken.setKind(tok::l_paren);
          TempToken.setLocation(ArgStartIterator->getLocation());
          TempToken.setLength(0);
          NewTokens.push_back(TempToken);
        }

        // Copy over argument tokens
        NewTokens.insert(NewTokens.end(), ArgStartIterator, I);

        // Add right paren and store the paren locations in ParenHints
        if (FoundSeparatorToken) {
          SourceLocation Loc = PP.getLocForEndOfToken((I - 1)->getLocation());
          TempToken.startToken();
          TempToken.setKind(tok::r_paren);
          TempToken.setLocation(Loc);
          TempToken.setLength(0);
          NewTokens.push_back(TempToken);
          ParenHints.push_back(SourceRange(ArgStartIterator->getLocation(),
                                           Loc));
        }

        // Copy separator token
        NewTokens.push_back(*I);

        // Reset values
        ArgStartIterator = I + 1;
        FoundSeparatorToken = false;
      }
    }
  }

  return !ParenHints.empty() && InitLists.empty();
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

  SourceLocation TooManyArgsLoc;

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
          return nullptr;
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
      } else if (Tok.is(tok::comma) && NumParens == 0 &&
                 !(Tok.getFlags() & Token::IgnoredComma)) {
        // In Microsoft-compatibility mode, single commas from nested macro
        // expansions should not be considered as argument separators. We test
        // for this with the IgnoredComma token flag above.

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
      } else if (Tok.getIdentifierInfo() != nullptr) {
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
    if (!isVariadic && NumFixedArgsLeft == 0 && TooManyArgsLoc.isInvalid()) {
      if (ArgTokens.size() != ArgTokenStart)
        TooManyArgsLoc = ArgTokens[ArgTokenStart].getLocation();
      else
        TooManyArgsLoc = ArgStartLoc;
    }

    // Empty arguments are standard in C99 and C++0x, and are supported as an
    // extension in other modes.
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
    if (!ContainsCodeCompletionTok && NumFixedArgsLeft != 0)
      --NumFixedArgsLeft;
  }

  // Okay, we either found the r_paren.  Check to see if we parsed too few
  // arguments.
  unsigned MinArgsExpected = MI->getNumArgs();

  // If this is not a variadic macro, and too many args were specified, emit
  // an error.
  if (!isVariadic && NumActuals > MinArgsExpected &&
      !ContainsCodeCompletionTok) {
    // Emit the diagnostic at the macro name in case there is a missing ).
    // Emitting it at the , could be far away from the macro name.
    Diag(TooManyArgsLoc, diag::err_too_many_args_in_macro_invoc);
    Diag(MI->getDefinitionLoc(), diag::note_macro_here)
      << MacroName.getIdentifierInfo();

    // Commas from braced initializer lists will be treated as argument
    // separators inside macros.  Attempt to correct for this with parentheses.
    // TODO: See if this can be generalized to angle brackets for templates
    // inside macro arguments.

    SmallVector<Token, 4> FixedArgTokens;
    unsigned FixedNumArgs = 0;
    SmallVector<SourceRange, 4> ParenHints, InitLists;
    if (!GenerateNewArgTokens(*this, ArgTokens, FixedArgTokens, FixedNumArgs,
                              ParenHints, InitLists)) {
      if (!InitLists.empty()) {
        DiagnosticBuilder DB =
            Diag(MacroName,
                 diag::note_init_list_at_beginning_of_macro_argument);
        for (const SourceRange &Range : InitLists)
          DB << Range;
      }
      return nullptr;
    }
    if (FixedNumArgs != MinArgsExpected)
      return nullptr;

    DiagnosticBuilder DB = Diag(MacroName, diag::note_suggest_parens_for_macro);
    for (const SourceRange &ParenLocation : ParenHints) {
      DB << FixItHint::CreateInsertion(ParenLocation.getBegin(), "(");
      DB << FixItHint::CreateInsertion(ParenLocation.getEnd(), ")");
    }
    ArgTokens.swap(FixedArgTokens);
    NumActuals = FixedNumArgs;
  }

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
      return nullptr;
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
    return nullptr;
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
    return nullptr;

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
      std::tie(prevLexer, tokIndex) = MacroExpandingLexersStack[i];
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
  time_t TT = time(nullptr);
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
           .Case("c_thread_safety_attributes", true)
           .Case("cxx_exceptions", LangOpts.CXXExceptions)
           .Case("cxx_rtti", LangOpts.RTTI)
           .Case("enumerator_attributes", true)
           .Case("memory_sanitizer", LangOpts.Sanitize.Memory)
           .Case("thread_sanitizer", LangOpts.Sanitize.Thread)
           .Case("dataflow_sanitizer", LangOpts.Sanitize.DataFlow)
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
           .Case("objc_protocol_qualifier_mangling", true)
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
           .Case("cxx_aggregate_nsdmi", LangOpts.CPlusPlus1y)
           .Case("cxx_binary_literals", LangOpts.CPlusPlus1y)
           .Case("cxx_contextual_conversions", LangOpts.CPlusPlus1y)
           .Case("cxx_decltype_auto", LangOpts.CPlusPlus1y)
           .Case("cxx_generic_lambdas", LangOpts.CPlusPlus1y)
           .Case("cxx_init_captures", LangOpts.CPlusPlus1y)
           .Case("cxx_relaxed_constexpr", LangOpts.CPlusPlus1y)
           .Case("cxx_return_type_deduction", LangOpts.CPlusPlus1y)
           .Case("cxx_variable_templates", LangOpts.CPlusPlus1y)
           // C++ TSes
           //.Case("cxx_runtime_arrays", LangOpts.CPlusPlusTSArrays)
           //.Case("cxx_concepts", LangOpts.CPlusPlusTSConcepts)
           // FIXME: Should this be __has_feature or __has_extension?
           //.Case("raw_invocation_type", LangOpts.CPlusPlus)
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
           .Case("is_constructible", LangOpts.CPlusPlus)
           .Case("is_convertible_to", LangOpts.CPlusPlus)
           .Case("is_empty", LangOpts.CPlusPlus)
           .Case("is_enum", LangOpts.CPlusPlus)
           .Case("is_final", LangOpts.CPlusPlus)
           .Case("is_literal", LangOpts.CPlusPlus)
           .Case("is_standard_layout", LangOpts.CPlusPlus)
           .Case("is_pod", LangOpts.CPlusPlus)
           .Case("is_polymorphic", LangOpts.CPlusPlus)
           .Case("is_sealed", LangOpts.MicrosoftExt)
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
  if (PP.getDiagnostics().getExtensionHandlingBehavior() >=
      diag::Severity::Error)
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
           .Case("c_thread_local", PP.getTargetInfo().isTLSSupported())
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
           .Case("cxx_init_captures", LangOpts.CPlusPlus11)
           .Case("cxx_variable_templates", LangOpts.CPlusPlus)
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
    PP.Diag(LParenLoc, diag::err_pp_expected_after) << II << tok::l_paren;
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
    PP.Diag(PP.getLocForEndOfToken(FilenameLoc), diag::err_pp_expected_after)
        << II << tok::r_paren;
    PP.Diag(LParenLoc, diag::note_matching) << tok::l_paren;
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
      PP.LookupFile(FilenameLoc, Filename, isAngled, LookupFrom, CurDir,
                    nullptr, nullptr, nullptr);

  // Get the result value.  A result of true means the file exists.
  return File != nullptr;
}

/// EvaluateHasInclude - Process a '__has_include("path")' expression.
/// Returns true if successful.
static bool EvaluateHasInclude(Token &Tok, IdentifierInfo *II,
                               Preprocessor &PP) {
  return EvaluateHasIncludeCommon(Tok, II, PP, nullptr);
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
    Lookup = nullptr;
    PP.Diag(Tok, diag::pp_include_next_in_primary);
  } else if (!Lookup) {
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
    PP.Diag(Tok.getLocation(), diag::err_pp_expected_after) << II
                                                            << tok::l_paren;
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
    PP.Diag(Tok.getLocation(), diag::err_pp_expected_after) << II
                                                            << tok::r_paren;
    PP.Diag(LParenLoc, diag::note_matching) << tok::l_paren;
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
  Tok.setIdentifierInfo(nullptr);
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
    Diag(Tok.getLocation(), diag::warn_pp_date_time);
    if (!DATELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"Mmm dd yyyy\""));
    Tok.setLocation(SourceMgr.createExpansionLoc(DATELoc, Tok.getLocation(),
                                                 Tok.getLocation(),
                                                 Tok.getLength()));
    return;
  } else if (II == Ident__TIME__) {
    Diag(Tok.getLocation(), diag::warn_pp_date_time);
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
    Diag(Tok.getLocation(), diag::warn_pp_date_time);
    // MSVC, ICC, GCC, VisualAge C++ extension.  The generated string should be
    // of the form "Ddd Mmm dd hh::mm::ss yyyy", which is returned by asctime.

    // Get the file that we are lexing out of.  If we're currently lexing from
    // a macro, dig into the include stack.
    const FileEntry *CurFile = nullptr;
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
    OS << '"' << StringRef(Result).drop_back() << '"';
    Tok.setKind(tok::string_literal);
  } else if (II == Ident__COUNTER__) {
    // __COUNTER__ expands to a simple numeric value.
    OS << CounterValue++;
    Tok.setKind(tok::numeric_constant);
  } else if (II == Ident__has_feature   ||
             II == Ident__has_extension ||
             II == Ident__has_builtin   ||
             II == Ident__is_identifier ||
             II == Ident__has_attribute) {
    // The argument to these builtins should be a parenthesized identifier.
    SourceLocation StartLoc = Tok.getLocation();

    bool IsValid = false;
    IdentifierInfo *FeatureII = nullptr;

    // Read the '('.
    LexUnexpandedToken(Tok);
    if (Tok.is(tok::l_paren)) {
      // Read the identifier
      LexUnexpandedToken(Tok);
      if ((FeatureII = Tok.getIdentifierInfo())) {
        // Read the ')'.
        LexUnexpandedToken(Tok);
        if (Tok.is(tok::r_paren))
          IsValid = true;
      }
    }

    bool Value = false;
    if (!IsValid)
      Diag(StartLoc, diag::err_feature_check_malformed);
    else if (II == Ident__is_identifier)
      Value = FeatureII->getTokenID() == tok::identifier;
    else if (II == Ident__has_builtin) {
      // Check for a builtin is trivial.
      Value = FeatureII->getBuiltinID() != 0;
    } else if (II == Ident__has_attribute)
      Value = hasAttribute(AttrSyntax::Generic, nullptr, FeatureII,
                           getTargetInfo().getTriple(), getLangOpts());
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

      // FIXME: Should we accept "-R..." flags here, or should that be handled
      // by a separate __has_remark?
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
        getDiagnosticsInGroup(diag::Flavor::WarningOrError,
                              WarningName.substr(2), Diags);
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
  } else if (II == Ident__identifier) {
    SourceLocation Loc = Tok.getLocation();

    // We're expecting '__identifier' '(' identifier ')'. Try to recover
    // if the parens are missing.
    LexNonComment(Tok);
    if (Tok.isNot(tok::l_paren)) {
      // No '(', use end of last token.
      Diag(getLocForEndOfToken(Loc), diag::err_pp_expected_after)
        << II << tok::l_paren;
      // If the next token isn't valid as our argument, we can't recover.
      if (!Tok.isAnnotation() && Tok.getIdentifierInfo())
        Tok.setKind(tok::identifier);
      return;
    }

    SourceLocation LParenLoc = Tok.getLocation();
    LexNonComment(Tok);

    if (!Tok.isAnnotation() && Tok.getIdentifierInfo())
      Tok.setKind(tok::identifier);
    else {
      Diag(Tok.getLocation(), diag::err_pp_identifier_arg_not_identifier)
        << Tok.getKind();
      // Don't walk past anything that's not a real token.
      if (Tok.is(tok::eof) || Tok.is(tok::eod) || Tok.isAnnotation())
        return;
    }

    // Discard the ')', preserving 'Tok' as our result.
    Token RParen;
    LexNonComment(RParen);
    if (RParen.isNot(tok::r_paren)) {
      Diag(getLocForEndOfToken(Tok.getLocation()), diag::err_pp_expected_after)
        << Tok.getKind() << tok::r_paren;
      Diag(LParenLoc, diag::note_matching) << tok::l_paren;
    }
    return;
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
