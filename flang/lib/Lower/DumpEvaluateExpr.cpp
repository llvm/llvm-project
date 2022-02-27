//===-- Lower/DumpEvaluateExpr.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/DumpEvaluateExpr.h"
#include <iostream>

static constexpr char whiteSpacePadding[] =
    ">>                                               ";
static constexpr auto whiteSize = sizeof(whiteSpacePadding) - 1;

inline const char *Fortran::lower::DumpEvaluateExpr::getIndentString() const {
  auto count = (level * 2 >= whiteSize) ? whiteSize : level * 2;
  return whiteSpacePadding + whiteSize - count;
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::CoarrayRef &x) {
  indent("coarray ref");
  show(x.base());
  show(x.subscript());
  show(x.cosubscript());
  show(x.stat());
  show(x.team());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::BOZLiteralConstant &) {
  print("BOZ literal constant");
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::NullPointer &) {
  print("null pointer");
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::semantics::Symbol &symbol) {
  const auto &ultimate{symbol.GetUltimate()};
  print("symbol: "s + std::string(toStringRef(symbol.name())));
  if (const auto *assoc =
          ultimate.detailsIf<Fortran::semantics::AssocEntityDetails>()) {
    indent("assoc details");
    show(assoc->expr());
    outdent();
  }
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::StaticDataObject &) {
  print("static data object");
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::ImpliedDoIndex &) {
  print("implied do index");
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::BaseObject &x) {
  indent("base object");
  show(x.u);
  outdent();
}
void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::Component &x) {
  indent("component");
  show(x.base());
  show(x.GetLastSymbol());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::NamedEntity &x) {
  indent("named entity");
  if (const auto *component = x.UnwrapComponent())
    show(*component);
  else
    show(x.GetFirstSymbol());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::TypeParamInquiry &x) {
  indent("type inquiry");
  show(x.base());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::Triplet &x) {
  indent("triplet");
  show(x.lower());
  show(x.upper());
  show(x.stride());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::Subscript &x) {
  indent("subscript");
  show(x.u);
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::ArrayRef &x) {
  indent("array ref");
  show(x.base());
  show(x.subscript());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::DataRef &x) {
  indent("data ref");
  show(x.u);
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::Substring &x) {
  indent("substring");
  show(x.parent());
  show(x.lower());
  show(x.upper());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::semantics::ParamValue &x) {
  indent("param value");
  show(x.GetExplicit());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::semantics::DerivedTypeSpec::ParameterMapType::value_type
        &x) {
  show(x.second);
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::semantics::DerivedTypeSpec &x) {
  indent("derived type spec");
  for (auto &v : x.parameters())
    show(v);
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::StructureConstructorValues::value_type &x) {
  show(x.second);
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::StructureConstructor &x) {
  indent("structure constructor");
  show(x.derivedTypeSpec());
  for (auto &v : x)
    show(v);
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x) {
  indent("expr some type");
  show(x.u);
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::ComplexPart &x) {
  indent("complex part");
  show(x.complex());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::ActualArgument &x) {
  indent("actual argument");
  if (const auto *symbol = x.GetAssumedTypeDummy())
    show(*symbol);
  else
    show(x.UnwrapExpr());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::ProcedureDesignator &x) {
  indent("procedure designator");
  if (const auto *component = x.GetComponent())
    show(*component);
  else if (const auto *symbol = x.GetSymbol())
    show(*symbol);
  else
    show(DEREF(x.GetSpecificIntrinsic()));
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::SpecificIntrinsic &) {
  print("specific intrinsic");
}

void Fortran::lower::DumpEvaluateExpr::show(
    const Fortran::evaluate::DescriptorInquiry &x) {
  indent("descriptor inquiry");
  show(x.base());
  outdent();
}

void Fortran::lower::DumpEvaluateExpr::print(llvm::Twine twine) {
  outs << getIndentString() << twine << '\n';
}

void Fortran::lower::DumpEvaluateExpr::indent(llvm::StringRef s) {
  print(s + " {");
  level++;
}

void Fortran::lower::DumpEvaluateExpr::outdent() {
  if (level)
    level--;
  print("}");
}

//===----------------------------------------------------------------------===//
// Boilerplate entry points that the debugger can find.
//===----------------------------------------------------------------------===//

void Fortran::lower::dumpEvExpr(const Fortran::semantics::SomeExpr &x) {
  DumpEvaluateExpr::dump(x);
}

void Fortran::lower::dumpEvExpr(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 4>>
        &x) {
  DumpEvaluateExpr::dump(x);
}

void Fortran::lower::dumpEvExpr(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>>
        &x) {
  DumpEvaluateExpr::dump(x);
}

void Fortran::lower::dumpEvExpr(const Fortran::evaluate::ArrayRef &x) {
  DumpEvaluateExpr::dump(x);
}

void Fortran::lower::dumpEvExpr(const Fortran::evaluate::DataRef &x) {
  DumpEvaluateExpr::dump(x);
}

void Fortran::lower::dumpEvExpr(const Fortran::evaluate::Substring &x) {
  DumpEvaluateExpr::dump(x);
}

void Fortran::lower::dumpEvExpr(
    const Fortran::evaluate::Designator<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 4>>
        &x) {
  DumpEvaluateExpr::dump(x);
}
