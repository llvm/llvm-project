//===--- ObjCPropertyAttributeOrderFixer.cpp -------------------*- C++--*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements ObjCPropertyAttributeOrderFixer, a TokenAnalyzer that
/// adjusts the order of attributes in an ObjC `@property(...)` declaration,
/// depending on the style.
///
//===----------------------------------------------------------------------===//

#include "ObjCPropertyAttributeOrderFixer.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

namespace clang {
namespace format {

ObjCPropertyAttributeOrderFixer::ObjCPropertyAttributeOrderFixer(
    const Environment &Env, const FormatStyle &Style)
    : TokenAnalyzer(Env, Style) {

  // Create an "order priority" map to use to sort properties.
  unsigned index = 0;
  for (const auto &Property : Style.ObjCPropertyAttributeOrder)
    SortOrderMap[Property] = index++;
}

struct ObjCPropertyEntry {
  StringRef Attribute; // eg, "readwrite"
  StringRef Value;     // eg, the "foo" of the attribute "getter=foo"
};

static bool isObjCPropertyAttribute(const FormatToken *Tok) {
  // Most attributes look like identifiers, but `class` is a keyword.
  return Tok->isOneOf(tok::identifier, tok::kw_class);
}

void ObjCPropertyAttributeOrderFixer::sortPropertyAttributes(
    const SourceManager &SourceMgr, tooling::Replacements &Fixes,
    const FormatToken *BeginTok, const FormatToken *EndTok) const {
  assert(BeginTok);
  assert(EndTok);
  assert(EndTok->Previous);

  // If there are zero or one tokens, nothing to do.
  if (BeginTok == EndTok || BeginTok->Next == EndTok)
    return;

  // Collect the attributes.
  SmallVector<ObjCPropertyEntry, 8> PropertyAttributes;
  for (auto Tok = BeginTok; Tok != EndTok; Tok = Tok->Next) {
    assert(Tok);
    if (Tok->is(tok::comma)) {
      // Ignore the comma separators.
      continue;
    }

    if (!isObjCPropertyAttribute(Tok)) {
      // If we hit any other kind of token, just bail.
      return;
    }

    // Memoize the attribute. (Note that 'class' is a legal attribute!)
    PropertyAttributes.push_back({Tok->TokenText, StringRef{}});

    // Also handle `getter=getFoo` attributes.
    // (Note: no check needed against `EndTok`, since its type is not
    // BinaryOperator or Identifier)
    assert(Tok->Next);
    if (Tok->Next->is(tok::equal)) {
      Tok = Tok->Next;
      assert(Tok->Next);
      if (Tok->Next->isNot(tok::identifier)) {
        // If we hit any other kind of token, just bail. It's unusual/illegal.
        return;
      }
      Tok = Tok->Next;
      PropertyAttributes.back().Value = Tok->TokenText;
    }
  }

  // There's nothing to do unless there's more than one attribute.
  if (PropertyAttributes.size() < 2)
    return;

  // Create a "remapping index" on how to reorder the attributes.
  SmallVector<unsigned, 8> Indices =
      llvm::to_vector<8>(llvm::seq<unsigned>(0, PropertyAttributes.size()));

  // Sort the indices based on the priority stored in 'SortOrderMap'; use Max
  // for missing values.
  const auto SortOrderMax = Style.ObjCPropertyAttributeOrder.size();
  auto SortIndex = [&](const StringRef &Needle) -> unsigned {
    auto I = SortOrderMap.find(Needle);
    return (I == SortOrderMap.end()) ? SortOrderMax : I->getValue();
  };
  llvm::stable_sort(Indices, [&](unsigned LHSI, unsigned RHSI) {
    return SortIndex(PropertyAttributes[LHSI].Attribute) <
           SortIndex(PropertyAttributes[RHSI].Attribute);
  });

  // If the property order is already correct, then no fix-up is needed.
  if (llvm::is_sorted(Indices))
    return;

  // Generate the replacement text.
  std::string NewText;
  const auto AppendAttribute = [&](const ObjCPropertyEntry &PropertyEntry) {
    NewText += PropertyEntry.Attribute;

    if (!PropertyEntry.Value.empty()) {
      NewText += "=";
      NewText += PropertyEntry.Value;
    }
  };

  AppendAttribute(PropertyAttributes[Indices[0]]);
  for (unsigned Index : llvm::drop_begin(Indices)) {
    NewText += ", ";
    AppendAttribute(PropertyAttributes[Index]);
  }

  auto Range = CharSourceRange::getCharRange(
      BeginTok->getStartOfNonWhitespace(), EndTok->Previous->Tok.getEndLoc());
  auto Replacement = tooling::Replacement(SourceMgr, Range, NewText);
  auto Err = Fixes.add(Replacement);
  if (Err) {
    llvm::errs() << "Error while reodering ObjC property attributes : "
                 << llvm::toString(std::move(Err)) << "\n";
  }
}

void ObjCPropertyAttributeOrderFixer::analyzeObjCPropertyDecl(
    const SourceManager &SourceMgr, const AdditionalKeywords &Keywords,
    tooling::Replacements &Fixes, const FormatToken *Tok) const {
  assert(Tok);

  // Expect `property` to be the very next token or else just bail early.
  const FormatToken *const PropertyTok = Tok->Next;
  if (!PropertyTok || PropertyTok->isNot(Keywords.kw_property))
    return;

  // Expect the opening paren to be the next token or else just bail early.
  const FormatToken *const LParenTok = PropertyTok->getNextNonComment();
  if (!LParenTok || LParenTok->isNot(tok::l_paren))
    return;

  // Get the matching right-paren, the bounds for property attributes.
  const FormatToken *const RParenTok = LParenTok->MatchingParen;
  if (!RParenTok)
    return;

  sortPropertyAttributes(SourceMgr, Fixes, LParenTok->Next, RParenTok);
}

std::pair<tooling::Replacements, unsigned>
ObjCPropertyAttributeOrderFixer::analyze(
    TokenAnnotator & /*Annotator*/,
    SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  tooling::Replacements Fixes;
  const AdditionalKeywords &Keywords = Tokens.getKeywords();
  const SourceManager &SourceMgr = Env.getSourceManager();
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);

  for (AnnotatedLine *Line : AnnotatedLines) {
    assert(Line);
    if (!Line->Affected || Line->Type != LT_ObjCProperty)
      continue;
    FormatToken *First = Line->First;
    assert(First);
    if (First->Finalized)
      continue;

    const auto *Last = Line->Last;

    for (const auto *Tok = First; Tok != Last; Tok = Tok->Next) {
      assert(Tok);

      // Skip until the `@` of a `@property` declaration.
      if (Tok->isNot(TT_ObjCProperty))
        continue;

      analyzeObjCPropertyDecl(SourceMgr, Keywords, Fixes, Tok);

      // There are never two `@property` in a line (they are split
      // by other passes), so this pass can break after just one.
      break;
    }
  }
  return {Fixes, 0};
}

} // namespace format
} // namespace clang
