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

using namespace clang;

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
