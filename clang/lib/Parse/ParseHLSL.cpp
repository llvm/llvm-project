//===--- ParseHLSL.cpp - HLSL-specific parsing support --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parsing logic for HLSL language features.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/AttributeCommonInfo.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"

using namespace clang;

static bool validateDeclsInsideHLSLBuffer(Parser::DeclGroupPtrTy DG,
                                          SourceLocation BufferLoc,
                                          bool IsCBuffer, Parser &P) {
  // The parse is failed, just return false.
  if (!DG)
    return false;
  DeclGroupRef Decls = DG.get();
  bool IsValid = true;
  // Only allow function, variable, record decls inside HLSLBuffer.
  for (DeclGroupRef::iterator I = Decls.begin(), E = Decls.end(); I != E; ++I) {
    Decl *D = *I;
    if (isa<CXXRecordDecl, RecordDecl, FunctionDecl, VarDecl>(D))
      continue;

    // FIXME: support nested HLSLBuffer and namespace inside HLSLBuffer.
    if (isa<HLSLBufferDecl, NamespaceDecl>(D)) {
      P.Diag(D->getLocation(), diag::err_invalid_declaration_in_hlsl_buffer)
          << IsCBuffer;
      IsValid = false;
      continue;
    }

    IsValid = false;
    P.Diag(D->getLocation(), diag::err_invalid_declaration_in_hlsl_buffer)
        << IsCBuffer;
  }
  return IsValid;
}

Decl *Parser::ParseHLSLBuffer(SourceLocation &DeclEnd) {
  assert((Tok.is(tok::kw_cbuffer) || Tok.is(tok::kw_tbuffer)) &&
         "Not a cbuffer or tbuffer!");
  bool IsCBuffer = Tok.is(tok::kw_cbuffer);
  SourceLocation BufferLoc = ConsumeToken(); // Eat the 'cbuffer' or 'tbuffer'.

  if (!Tok.is(tok::identifier)) {
    Diag(Tok, diag::err_expected) << tok::identifier;
    return nullptr;
  }

  IdentifierInfo *Identifier = Tok.getIdentifierInfo();
  SourceLocation IdentifierLoc = ConsumeToken();

  ParseScope BufferScope(this, Scope::DeclScope);
  BalancedDelimiterTracker T(*this, tok::l_brace);
  if (T.consumeOpen()) {
    Diag(Tok, diag::err_expected) << tok::l_brace;
    return nullptr;
  }

  Decl *D = Actions.ActOnStartHLSLBuffer(getCurScope(), IsCBuffer, BufferLoc,
                                         Identifier, IdentifierLoc,
                                         T.getOpenLocation());

  // FIXME: support attribute on cbuffer/tbuffer.
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    SourceLocation Loc = Tok.getLocation();
    // FIXME: support attribute on constants inside cbuffer/tbuffer.
    ParsedAttributes Attrs(AttrFactory);

    DeclGroupPtrTy Result = ParseExternalDeclaration(Attrs);
    if (!validateDeclsInsideHLSLBuffer(Result, IdentifierLoc, IsCBuffer,
                                       *this)) {
      T.skipToEnd();
      DeclEnd = T.getCloseLocation();
      BufferScope.Exit();
      Actions.ActOnFinishHLSLBuffer(D, DeclEnd);
      return nullptr;
    }
  }

  T.consumeClose();
  DeclEnd = T.getCloseLocation();
  BufferScope.Exit();
  Actions.ActOnFinishHLSLBuffer(D, DeclEnd);

  return D;
}

void Parser::ParseHLSLSemantics(ParsedAttributes &Attrs,
                                SourceLocation *EndLoc) {
  assert(Tok.is(tok::colon) && "Not a HLSL Semantic");
  ConsumeToken();

  if (!Tok.is(tok::identifier)) {
    Diag(Tok.getLocation(), diag::err_expected_semantic_identifier);
    return;
  }

  IdentifierInfo *II = Tok.getIdentifierInfo();
  SourceLocation Loc = ConsumeToken();
  if (EndLoc)
    *EndLoc = Tok.getLocation();
  ParsedAttr::Kind AttrKind =
      ParsedAttr::getParsedKind(II, nullptr, ParsedAttr::AS_HLSLSemantic);

  if (AttrKind == ParsedAttr::UnknownAttribute) {
    Diag(Loc, diag::err_unknown_hlsl_semantic) << II;
    return;
  }
  Attrs.addNew(II, Loc, nullptr, SourceLocation(), nullptr, 0,
               ParsedAttr::AS_HLSLSemantic);
}
