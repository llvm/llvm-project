//===-- Semantics/dump-expr.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/dump-expr.h"

namespace Fortran::semantics {

static constexpr char whiteSpacePadding[]{
    ">>                                               "};
static constexpr auto whiteSize{sizeof(whiteSpacePadding) - 1};

inline const char *DumpEvaluateExpr::GetIndentString() const {
  auto count{(level_ * 2 >= whiteSize) ? whiteSize : level_ * 2};
  return whiteSpacePadding + whiteSize - count;
}

void DumpEvaluateExpr::Show(const evaluate::CoarrayRef &x) {
  Indent("coarray ref");
  Show(x.base());
  Show(x.cosubscript());
  Show(x.stat());
  Show(x.team());
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::BOZLiteralConstant &) {
  Print("BOZ literal constant");
}

void DumpEvaluateExpr::Show(const evaluate::NullPointer &) {
  Print("null pointer");
}

void DumpEvaluateExpr::Show(const Symbol &symbol) {
  const auto &ultimate{symbol.GetUltimate()};
  Print("symbol: "s + symbol.name().ToString());
  if (const auto *assoc{ultimate.detailsIf<AssocEntityDetails>()}) {
    Indent("assoc details");
    Show(assoc->expr());
    Outdent();
  }
}

void DumpEvaluateExpr::Show(const evaluate::StaticDataObject &) {
  Print("static data object");
}

void DumpEvaluateExpr::Show(const evaluate::ImpliedDoIndex &) {
  Print("implied do index");
}

void DumpEvaluateExpr::Show(const evaluate::BaseObject &x) {
  Indent("base object");
  Show(x.u);
  Outdent();
}
void DumpEvaluateExpr::Show(const evaluate::Component &x) {
  Indent("component");
  Show(x.base());
  Show(x.GetLastSymbol());
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::NamedEntity &x) {
  Indent("named entity");
  if (const auto *component{x.UnwrapComponent()}) {
    Show(*component);
  } else {
    Show(x.GetFirstSymbol());
  }
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::TypeParamInquiry &x) {
  Indent("type inquiry");
  Show(x.base());
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::Triplet &x) {
  Indent("triplet");
  Show(x.lower());
  Show(x.upper());
  Show(x.stride());
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::Subscript &x) {
  Indent("subscript");
  Show(x.u);
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::ArrayRef &x) {
  Indent("array ref");
  Show(x.base());
  Show(x.subscript());
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::DataRef &x) {
  Indent("data ref");
  Show(x.u);
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::Substring &x) {
  Indent("substring");
  Show(x.parent());
  Show(x.lower());
  Show(x.upper());
  Outdent();
}

void DumpEvaluateExpr::Show(const ParamValue &x) {
  Indent("param value");
  Show(x.GetExplicit());
  Outdent();
}

void DumpEvaluateExpr::Show(
    const DerivedTypeSpec::ParameterMapType::value_type &x) {
  Show(x.second);
}

void DumpEvaluateExpr::Show(const DerivedTypeSpec &x) {
  Indent("derived type spec");
  for (auto &v : x.parameters()) {
    Show(v);
  }
  Outdent();
}

void DumpEvaluateExpr::Show(
    const evaluate::StructureConstructorValues::value_type &x) {
  Show(x.second);
}

void DumpEvaluateExpr::Show(const evaluate::StructureConstructor &x) {
  Indent("structure constructor");
  Show(x.derivedTypeSpec());
  for (auto &v : x) {
    Show(v);
  }
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::Relational<evaluate::SomeType> &x) {
  Indent("relational some type");
  Show(x.u);
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::ComplexPart &x) {
  Indent("complex part");
  Show(x.complex());
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::ActualArgument &x) {
  Indent("actual argument");
  if (const auto *symbol{x.GetAssumedTypeDummy()}) {
    Show(*symbol);
  } else {
    Show(x.UnwrapExpr());
  }
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::ProcedureDesignator &x) {
  Indent("procedure designator");
  if (const auto *component{x.GetComponent()}) {
    Show(*component);
  } else if (const auto *symbol{x.GetSymbol()}) {
    Show(*symbol);
  } else {
    Show(DEREF(x.GetSpecificIntrinsic()));
  }
  Outdent();
}

void DumpEvaluateExpr::Show(const evaluate::SpecificIntrinsic &) {
  Print("specific intrinsic");
}

void DumpEvaluateExpr::Show(const evaluate::DescriptorInquiry &x) {
  Indent("descriptor inquiry");
  Show(x.base());
  Outdent();
}

void DumpEvaluateExpr::Print(llvm::Twine twine) {
  outs_ << GetIndentString() << twine << '\n';
}

void DumpEvaluateExpr::Indent(llvm::StringRef s) {
  Print(s + " {");
  level_++;
}

void DumpEvaluateExpr::Outdent() {
  if (level_) {
    level_--;
  }
  Print("}");
}

//===----------------------------------------------------------------------===//
// Boilerplate entry points that the debugger can find.
//===----------------------------------------------------------------------===//

void DumpEvExpr(const SomeExpr &x) { DumpEvaluateExpr::Dump(x); }

void DumpEvExpr(
    const evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 4>> &x) {
  DumpEvaluateExpr::Dump(x);
}

void DumpEvExpr(
    const evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 8>> &x) {
  DumpEvaluateExpr::Dump(x);
}

void DumpEvExpr(const evaluate::ArrayRef &x) { DumpEvaluateExpr::Dump(x); }

void DumpEvExpr(const evaluate::DataRef &x) { DumpEvaluateExpr::Dump(x); }

void DumpEvExpr(const evaluate::Substring &x) { DumpEvaluateExpr::Dump(x); }

void DumpEvExpr(
    const evaluate::Designator<evaluate::Type<common::TypeCategory::Integer, 4>>
        &x) {
  DumpEvaluateExpr::Dump(x);
}

} // namespace Fortran::semantics
