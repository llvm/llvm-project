//===--- ParsePragma.cpp - Language specific pragma parsing ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the language specific #pragma handlers.
//
//===----------------------------------------------------------------------===//

#include "ParsePragma.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang;

/// \brief Handle the annotation token produced for #pragma unused(...)
///
/// Each annot_pragma_unused is followed by the argument token so e.g.
/// "#pragma unused(x,y)" becomes:
/// annot_pragma_unused 'x' annot_pragma_unused 'y'
void Parser::HandlePragmaUnused() {
  assert(Tok.is(tok::annot_pragma_unused));
  SourceLocation UnusedLoc = ConsumeToken();
  Actions.ActOnPragmaUnused(Tok, getCurScope(), UnusedLoc);
  ConsumeToken(); // The argument token.
}

void Parser::HandlePragmaVisibility() {
  assert(Tok.is(tok::annot_pragma_vis));
  const IdentifierInfo *VisType =
    static_cast<IdentifierInfo *>(Tok.getAnnotationValue());
  SourceLocation VisLoc = ConsumeToken();
  Actions.ActOnPragmaVisibility(VisType, VisLoc);
}

struct PragmaPackInfo {
  Sema::PragmaPackKind Kind;
  IdentifierInfo *Name;
  Token Alignment;
  SourceLocation LParenLoc;
  SourceLocation RParenLoc;
};

void Parser::HandlePragmaPack() {
  assert(Tok.is(tok::annot_pragma_pack));
  PragmaPackInfo *Info =
    static_cast<PragmaPackInfo *>(Tok.getAnnotationValue());
  SourceLocation PragmaLoc = ConsumeToken();
  ExprResult Alignment;
  if (Info->Alignment.is(tok::numeric_constant)) {
    Alignment = Actions.ActOnNumericConstant(Info->Alignment);
    if (Alignment.isInvalid())
      return;
  }
  Actions.ActOnPragmaPack(Info->Kind, Info->Name, Alignment.get(), PragmaLoc,
                          Info->LParenLoc, Info->RParenLoc);
}

void Parser::HandlePragmaMSStruct() {
  assert(Tok.is(tok::annot_pragma_msstruct));
  Sema::PragmaMSStructKind Kind =
    static_cast<Sema::PragmaMSStructKind>(
    reinterpret_cast<uintptr_t>(Tok.getAnnotationValue()));
  Actions.ActOnPragmaMSStruct(Kind);
  ConsumeToken(); // The annotation token.
}

void Parser::HandlePragmaAlign() {
  assert(Tok.is(tok::annot_pragma_align));
  Sema::PragmaOptionsAlignKind Kind =
    static_cast<Sema::PragmaOptionsAlignKind>(
    reinterpret_cast<uintptr_t>(Tok.getAnnotationValue()));
  SourceLocation PragmaLoc = ConsumeToken();
  Actions.ActOnPragmaOptionsAlign(Kind, PragmaLoc);
}

void Parser::HandlePragmaWeak() {
  assert(Tok.is(tok::annot_pragma_weak));
  SourceLocation PragmaLoc = ConsumeToken();
  Actions.ActOnPragmaWeakID(Tok.getIdentifierInfo(), PragmaLoc,
                            Tok.getLocation());
  ConsumeToken(); // The weak name.
}

void Parser::HandlePragmaWeakAlias() {
  assert(Tok.is(tok::annot_pragma_weakalias));
  SourceLocation PragmaLoc = ConsumeToken();
  IdentifierInfo *WeakName = Tok.getIdentifierInfo();
  SourceLocation WeakNameLoc = Tok.getLocation();
  ConsumeToken();
  IdentifierInfo *AliasName = Tok.getIdentifierInfo();
  SourceLocation AliasNameLoc = Tok.getLocation();
  ConsumeToken();
  Actions.ActOnPragmaWeakAlias(WeakName, AliasName, PragmaLoc,
                               WeakNameLoc, AliasNameLoc);

}

void Parser::HandlePragmaRedefineExtname() {
  assert(Tok.is(tok::annot_pragma_redefine_extname));
  SourceLocation RedefLoc = ConsumeToken();
  IdentifierInfo *RedefName = Tok.getIdentifierInfo();
  SourceLocation RedefNameLoc = Tok.getLocation();
  ConsumeToken();
  IdentifierInfo *AliasName = Tok.getIdentifierInfo();
  SourceLocation AliasNameLoc = Tok.getLocation();
  ConsumeToken();
  Actions.ActOnPragmaRedefineExtname(RedefName, AliasName, RedefLoc,
                                     RedefNameLoc, AliasNameLoc);
}

void Parser::HandlePragmaFPContract() {
  assert(Tok.is(tok::annot_pragma_fp_contract));
  tok::OnOffSwitch OOS =
    static_cast<tok::OnOffSwitch>(
    reinterpret_cast<uintptr_t>(Tok.getAnnotationValue()));
  Actions.ActOnPragmaFPContract(OOS);
  ConsumeToken(); // The annotation token.
}

namespace {
  typedef llvm::PointerIntPair<IdentifierInfo *, 1, bool> OpenCLExtData;
}

void Parser::HandlePragmaOpenCLExtension() {
  assert(Tok.is(tok::annot_pragma_opencl_extension));
  OpenCLExtData data =
      OpenCLExtData::getFromOpaqueValue(Tok.getAnnotationValue());
  unsigned state = data.getInt();
  IdentifierInfo *ename = data.getPointer();
  SourceLocation NameLoc = Tok.getLocation();
  ConsumeToken(); // The annotation token.

  OpenCLOptions &f = Actions.getOpenCLOptions();
  // OpenCL 1.1 9.1: "The all variant sets the behavior for all extensions,
  // overriding all previously issued extension directives, but only if the
  // behavior is set to disable."
  if (state == 0 && ename->isStr("all")) {
#define OPENCLEXT(nm)   f.nm = 0;
#include "clang/Basic/OpenCLExtensions.def"
  }
#define OPENCLEXT(nm) else if (ename->isStr(#nm)) { f.nm = state; }
#include "clang/Basic/OpenCLExtensions.def"
  else {
    PP.Diag(NameLoc, diag::warn_pragma_unknown_extension) << ename;
    return;
  }
}

// #pragma GCC visibility comes in two variants:
//   'push' '(' [visibility] ')'
//   'pop'
void PragmaGCCVisibilityHandler::HandlePragma(Preprocessor &PP, 
                                              PragmaIntroducerKind Introducer,
                                              Token &VisTok) {
  SourceLocation VisLoc = VisTok.getLocation();

  Token Tok;
  PP.LexUnexpandedToken(Tok);

  const IdentifierInfo *PushPop = Tok.getIdentifierInfo();

  const IdentifierInfo *VisType;
  if (PushPop && PushPop->isStr("pop")) {
    VisType = 0;
  } else if (PushPop && PushPop->isStr("push")) {
    PP.LexUnexpandedToken(Tok);
    if (Tok.isNot(tok::l_paren)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen)
        << "visibility";
      return;
    }
    PP.LexUnexpandedToken(Tok);
    VisType = Tok.getIdentifierInfo();
    if (!VisType) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
        << "visibility";
      return;
    }
    PP.LexUnexpandedToken(Tok);
    if (Tok.isNot(tok::r_paren)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_rparen)
        << "visibility";
      return;
    }
  } else {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
      << "visibility";
    return;
  }
  PP.LexUnexpandedToken(Tok);
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
      << "visibility";
    return;
  }

  Token *Toks = new Token[1];
  Toks[0].startToken();
  Toks[0].setKind(tok::annot_pragma_vis);
  Toks[0].setLocation(VisLoc);
  Toks[0].setAnnotationValue(
                          const_cast<void*>(static_cast<const void*>(VisType)));
  PP.EnterTokenStream(Toks, 1, /*DisableMacroExpansion=*/true,
                      /*OwnsTokens=*/true);
}

// #pragma pack(...) comes in the following delicious flavors:
//   pack '(' [integer] ')'
//   pack '(' 'show' ')'
//   pack '(' ('push' | 'pop') [',' identifier] [, integer] ')'
void PragmaPackHandler::HandlePragma(Preprocessor &PP, 
                                     PragmaIntroducerKind Introducer,
                                     Token &PackTok) {
  SourceLocation PackLoc = PackTok.getLocation();

  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen) << "pack";
    return;
  }

  Sema::PragmaPackKind Kind = Sema::PPK_Default;
  IdentifierInfo *Name = 0;
  Token Alignment;
  Alignment.startToken();
  SourceLocation LParenLoc = Tok.getLocation();
  PP.Lex(Tok);
  if (Tok.is(tok::numeric_constant)) {
    Alignment = Tok;

    PP.Lex(Tok);

    // In MSVC/gcc, #pragma pack(4) sets the alignment without affecting
    // the push/pop stack.
    // In Apple gcc, #pragma pack(4) is equivalent to #pragma pack(push, 4)
    if (PP.getLangOpts().ApplePragmaPack)
      Kind = Sema::PPK_Push;
  } else if (Tok.is(tok::identifier)) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    if (II->isStr("show")) {
      Kind = Sema::PPK_Show;
      PP.Lex(Tok);
    } else {
      if (II->isStr("push")) {
        Kind = Sema::PPK_Push;
      } else if (II->isStr("pop")) {
        Kind = Sema::PPK_Pop;
      } else {
        PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_invalid_action);
        return;
      }
      PP.Lex(Tok);

      if (Tok.is(tok::comma)) {
        PP.Lex(Tok);

        if (Tok.is(tok::numeric_constant)) {
          Alignment = Tok;

          PP.Lex(Tok);
        } else if (Tok.is(tok::identifier)) {
          Name = Tok.getIdentifierInfo();
          PP.Lex(Tok);

          if (Tok.is(tok::comma)) {
            PP.Lex(Tok);

            if (Tok.isNot(tok::numeric_constant)) {
              PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_malformed);
              return;
            }

            Alignment = Tok;

            PP.Lex(Tok);
          }
        } else {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_malformed);
          return;
        }
      }
    }
  } else if (PP.getLangOpts().ApplePragmaPack) {
    // In MSVC/gcc, #pragma pack() resets the alignment without affecting
    // the push/pop stack.
    // In Apple gcc #pragma pack() is equivalent to #pragma pack(pop).
    Kind = Sema::PPK_Pop;
  }

  if (Tok.isNot(tok::r_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_rparen) << "pack";
    return;
  }

  SourceLocation RParenLoc = Tok.getLocation();
  PP.Lex(Tok);
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) << "pack";
    return;
  }

  PragmaPackInfo *Info = 
    (PragmaPackInfo*) PP.getPreprocessorAllocator().Allocate(
      sizeof(PragmaPackInfo), llvm::alignOf<PragmaPackInfo>());
  new (Info) PragmaPackInfo();
  Info->Kind = Kind;
  Info->Name = Name;
  Info->Alignment = Alignment;
  Info->LParenLoc = LParenLoc;
  Info->RParenLoc = RParenLoc;

  Token *Toks = 
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 1, llvm::alignOf<Token>());
  new (Toks) Token();
  Toks[0].startToken();
  Toks[0].setKind(tok::annot_pragma_pack);
  Toks[0].setLocation(PackLoc);
  Toks[0].setAnnotationValue(static_cast<void*>(Info));
  PP.EnterTokenStream(Toks, 1, /*DisableMacroExpansion=*/true,
                      /*OwnsTokens=*/false);
}

// #pragma ms_struct on
// #pragma ms_struct off
void PragmaMSStructHandler::HandlePragma(Preprocessor &PP, 
                                         PragmaIntroducerKind Introducer,
                                         Token &MSStructTok) {
  Sema::PragmaMSStructKind Kind = Sema::PMSST_OFF;
  
  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_ms_struct);
    return;
  }
  const IdentifierInfo *II = Tok.getIdentifierInfo();
  if (II->isStr("on")) {
    Kind = Sema::PMSST_ON;
    PP.Lex(Tok);
  }
  else if (II->isStr("off") || II->isStr("reset"))
    PP.Lex(Tok);
  else {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_ms_struct);
    return;
  }
  
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
      << "ms_struct";
    return;
  }

  Token *Toks =
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 1, llvm::alignOf<Token>());
  new (Toks) Token();
  Toks[0].startToken();
  Toks[0].setKind(tok::annot_pragma_msstruct);
  Toks[0].setLocation(MSStructTok.getLocation());
  Toks[0].setAnnotationValue(reinterpret_cast<void*>(
                             static_cast<uintptr_t>(Kind)));
  PP.EnterTokenStream(Toks, 1, /*DisableMacroExpansion=*/true,
                      /*OwnsTokens=*/false);
}

// #pragma 'align' '=' {'native','natural','mac68k','power','reset'}
// #pragma 'options 'align' '=' {'native','natural','mac68k','power','reset'}
static void ParseAlignPragma(Preprocessor &PP, Token &FirstTok,
                             bool IsOptions) {
  Token Tok;

  if (IsOptions) {
    PP.Lex(Tok);
    if (Tok.isNot(tok::identifier) ||
        !Tok.getIdentifierInfo()->isStr("align")) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_options_expected_align);
      return;
    }
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::equal)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_align_expected_equal)
      << IsOptions;
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
      << (IsOptions ? "options" : "align");
    return;
  }

  Sema::PragmaOptionsAlignKind Kind = Sema::POAK_Natural;
  const IdentifierInfo *II = Tok.getIdentifierInfo();
  if (II->isStr("native"))
    Kind = Sema::POAK_Native;
  else if (II->isStr("natural"))
    Kind = Sema::POAK_Natural;
  else if (II->isStr("packed"))
    Kind = Sema::POAK_Packed;
  else if (II->isStr("power"))
    Kind = Sema::POAK_Power;
  else if (II->isStr("mac68k"))
    Kind = Sema::POAK_Mac68k;
  else if (II->isStr("reset"))
    Kind = Sema::POAK_Reset;
  else {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_align_invalid_option)
      << IsOptions;
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
      << (IsOptions ? "options" : "align");
    return;
  }

  Token *Toks =
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 1, llvm::alignOf<Token>());
  new (Toks) Token();
  Toks[0].startToken();
  Toks[0].setKind(tok::annot_pragma_align);
  Toks[0].setLocation(FirstTok.getLocation());
  Toks[0].setAnnotationValue(reinterpret_cast<void*>(
                             static_cast<uintptr_t>(Kind)));
  PP.EnterTokenStream(Toks, 1, /*DisableMacroExpansion=*/true,
                      /*OwnsTokens=*/false);
}

void PragmaAlignHandler::HandlePragma(Preprocessor &PP, 
                                      PragmaIntroducerKind Introducer,
                                      Token &AlignTok) {
  ParseAlignPragma(PP, AlignTok, /*IsOptions=*/false);
}

void PragmaOptionsHandler::HandlePragma(Preprocessor &PP, 
                                        PragmaIntroducerKind Introducer,
                                        Token &OptionsTok) {
  ParseAlignPragma(PP, OptionsTok, /*IsOptions=*/true);
}

// #pragma unused(identifier)
void PragmaUnusedHandler::HandlePragma(Preprocessor &PP, 
                                       PragmaIntroducerKind Introducer,
                                       Token &UnusedTok) {
  // FIXME: Should we be expanding macros here? My guess is no.
  SourceLocation UnusedLoc = UnusedTok.getLocation();

  // Lex the left '('.
  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen) << "unused";
    return;
  }

  // Lex the declaration reference(s).
  SmallVector<Token, 5> Identifiers;
  SourceLocation RParenLoc;
  bool LexID = true;

  while (true) {
    PP.Lex(Tok);

    if (LexID) {
      if (Tok.is(tok::identifier)) {
        Identifiers.push_back(Tok);
        LexID = false;
        continue;
      }

      // Illegal token!
      PP.Diag(Tok.getLocation(), diag::warn_pragma_unused_expected_var);
      return;
    }

    // We are execting a ')' or a ','.
    if (Tok.is(tok::comma)) {
      LexID = true;
      continue;
    }

    if (Tok.is(tok::r_paren)) {
      RParenLoc = Tok.getLocation();
      break;
    }

    // Illegal token!
    PP.Diag(Tok.getLocation(), diag::warn_pragma_unused_expected_punc);
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) <<
        "unused";
    return;
  }

  // Verify that we have a location for the right parenthesis.
  assert(RParenLoc.isValid() && "Valid '#pragma unused' must have ')'");
  assert(!Identifiers.empty() && "Valid '#pragma unused' must have arguments");

  // For each identifier token, insert into the token stream a
  // annot_pragma_unused token followed by the identifier token.
  // This allows us to cache a "#pragma unused" that occurs inside an inline
  // C++ member function.

  Token *Toks = 
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 2 * Identifiers.size(), llvm::alignOf<Token>());
  for (unsigned i=0; i != Identifiers.size(); i++) {
    Token &pragmaUnusedTok = Toks[2*i], &idTok = Toks[2*i+1];
    pragmaUnusedTok.startToken();
    pragmaUnusedTok.setKind(tok::annot_pragma_unused);
    pragmaUnusedTok.setLocation(UnusedLoc);
    idTok = Identifiers[i];
  }
  PP.EnterTokenStream(Toks, 2*Identifiers.size(),
                      /*DisableMacroExpansion=*/true, /*OwnsTokens=*/false);
}

// #pragma weak identifier
// #pragma weak identifier '=' identifier
void PragmaWeakHandler::HandlePragma(Preprocessor &PP, 
                                     PragmaIntroducerKind Introducer,
                                     Token &WeakTok) {
  SourceLocation WeakLoc = WeakTok.getLocation();

  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier) << "weak";
    return;
  }

  Token WeakName = Tok;
  bool HasAlias = false;
  Token AliasName;

  PP.Lex(Tok);
  if (Tok.is(tok::equal)) {
    HasAlias = true;
    PP.Lex(Tok);
    if (Tok.isNot(tok::identifier)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
          << "weak";
      return;
    }
    AliasName = Tok;
    PP.Lex(Tok);
  }

  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) << "weak";
    return;
  }

  if (HasAlias) {
    Token *Toks = 
      (Token*) PP.getPreprocessorAllocator().Allocate(
        sizeof(Token) * 3, llvm::alignOf<Token>());
    Token &pragmaUnusedTok = Toks[0];
    pragmaUnusedTok.startToken();
    pragmaUnusedTok.setKind(tok::annot_pragma_weakalias);
    pragmaUnusedTok.setLocation(WeakLoc);
    Toks[1] = WeakName;
    Toks[2] = AliasName;
    PP.EnterTokenStream(Toks, 3,
                        /*DisableMacroExpansion=*/true, /*OwnsTokens=*/false);
  } else {
    Token *Toks = 
      (Token*) PP.getPreprocessorAllocator().Allocate(
        sizeof(Token) * 2, llvm::alignOf<Token>());
    Token &pragmaUnusedTok = Toks[0];
    pragmaUnusedTok.startToken();
    pragmaUnusedTok.setKind(tok::annot_pragma_weak);
    pragmaUnusedTok.setLocation(WeakLoc);
    Toks[1] = WeakName;
    PP.EnterTokenStream(Toks, 2,
                        /*DisableMacroExpansion=*/true, /*OwnsTokens=*/false);
  }
}

// #pragma redefine_extname identifier identifier
void PragmaRedefineExtnameHandler::HandlePragma(Preprocessor &PP, 
                                               PragmaIntroducerKind Introducer,
                                                Token &RedefToken) {
  SourceLocation RedefLoc = RedefToken.getLocation();

  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier) <<
      "redefine_extname";
    return;
  }

  Token RedefName = Tok;
  PP.Lex(Tok);

  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
        << "redefine_extname";
    return;
  }

  Token AliasName = Tok;
  PP.Lex(Tok);

  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) <<
      "redefine_extname";
    return;
  }

  Token *Toks = 
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 3, llvm::alignOf<Token>());
  Token &pragmaRedefTok = Toks[0];
  pragmaRedefTok.startToken();
  pragmaRedefTok.setKind(tok::annot_pragma_redefine_extname);
  pragmaRedefTok.setLocation(RedefLoc);
  Toks[1] = RedefName;
  Toks[2] = AliasName;
  PP.EnterTokenStream(Toks, 3,
                      /*DisableMacroExpansion=*/true, /*OwnsTokens=*/false);
}


void
PragmaFPContractHandler::HandlePragma(Preprocessor &PP, 
                                      PragmaIntroducerKind Introducer,
                                      Token &Tok) {
  tok::OnOffSwitch OOS;
  if (PP.LexOnOffSwitch(OOS))
    return;

  Token *Toks =
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 1, llvm::alignOf<Token>());
  new (Toks) Token();
  Toks[0].startToken();
  Toks[0].setKind(tok::annot_pragma_fp_contract);
  Toks[0].setLocation(Tok.getLocation());
  Toks[0].setAnnotationValue(reinterpret_cast<void*>(
                             static_cast<uintptr_t>(OOS)));
  PP.EnterTokenStream(Toks, 1, /*DisableMacroExpansion=*/true,
                      /*OwnsTokens=*/false);
}

void 
PragmaOpenCLExtensionHandler::HandlePragma(Preprocessor &PP, 
                                           PragmaIntroducerKind Introducer,
                                           Token &Tok) {
  PP.LexUnexpandedToken(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier) <<
      "OPENCL";
    return;
  }
  IdentifierInfo *ename = Tok.getIdentifierInfo();
  SourceLocation NameLoc = Tok.getLocation();

  PP.Lex(Tok);
  if (Tok.isNot(tok::colon)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_colon) << ename;
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_enable_disable);
    return;
  }
  IdentifierInfo *op = Tok.getIdentifierInfo();

  unsigned state;
  if (op->isStr("enable")) {
    state = 1;
  } else if (op->isStr("disable")) {
    state = 0;
  } else {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_enable_disable);
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::eod)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) <<
      "OPENCL EXTENSION";
    return;
  }

  OpenCLExtData data(ename, state);
  Token *Toks =
    (Token*) PP.getPreprocessorAllocator().Allocate(
      sizeof(Token) * 1, llvm::alignOf<Token>());
  new (Toks) Token();
  Toks[0].startToken();
  Toks[0].setKind(tok::annot_pragma_opencl_extension);
  Toks[0].setLocation(NameLoc);
  Toks[0].setAnnotationValue(data.getOpaqueValue());
  PP.EnterTokenStream(Toks, 1, /*DisableMacroExpansion=*/true,
                      /*OwnsTokens=*/false);
}

