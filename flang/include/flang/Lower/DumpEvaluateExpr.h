//===-- Lower/DumpEvaluateExpr.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_DUMPEVALUATEEXPR_H
#define FORTRAN_LOWER_DUMPEVALUATEEXPR_H

#include "flang/Evaluate/tools.h"
#include "flang/Lower/Support/Utils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Fortran::lower {

/// Class to dump Fortran::evaluate::Expr trees out in a user readable way.
///
/// FIXME: This can be improved to dump more information in some cases.
class DumpEvaluateExpr {
public:
  DumpEvaluateExpr() : outs(llvm::errs()) {}
  DumpEvaluateExpr(llvm::raw_ostream &str) : outs(str) {}

  template <typename A>
  static void dump(const A &x) {
    DumpEvaluateExpr{}.show(x);
  }
  template <typename A>
  static void dump(llvm::raw_ostream &stream, const A &x) {
    DumpEvaluateExpr{stream}.show(x);
  }

private:
  template <typename A, bool C>
  void show(const Fortran::common::Indirection<A, C> &x) {
    show(x.value());
  }
  template <typename A>
  void show(const Fortran::semantics::SymbolRef x) {
    show(*x);
  }
  template <typename A>
  void show(const std::unique_ptr<A> &x) {
    show(x.get());
  }
  template <typename A>
  void show(const std::shared_ptr<A> &x) {
    show(x.get());
  }
  template <typename A>
  void show(const A *x) {
    if (x) {
      show(*x);
      return;
    }
    print("nullptr");
  }
  template <typename A>
  void show(const std::optional<A> &x) {
    if (x) {
      show(*x);
      return;
    }
    print("None");
  }
  template <typename... A>
  void show(const std::variant<A...> &u) {
    std::visit([&](const auto &v) { show(v); }, u);
  }
  template <typename A>
  void show(const std::vector<A> &x) {
    indent("vector");
    for (const auto &v : x)
      show(v);
    outdent();
  }
  void show(const Fortran::evaluate::BOZLiteralConstant &);
  void show(const Fortran::evaluate::NullPointer &);
  template <typename T>
  void show(const Fortran::evaluate::Constant<T> &x) {
    if constexpr (T::category == Fortran::common::TypeCategory::Derived) {
      indent("derived constant");
      for (const auto &map : x.values())
        for (const auto &pair : map)
          show(pair.second.value());
      outdent();
    } else {
      print("constant");
    }
  }
  void show(const Fortran::semantics::Symbol &symbol);
  void show(const Fortran::evaluate::StaticDataObject &);
  void show(const Fortran::evaluate::ImpliedDoIndex &);
  void show(const Fortran::evaluate::BaseObject &x);
  void show(const Fortran::evaluate::Component &x);
  void show(const Fortran::evaluate::NamedEntity &x);
  void show(const Fortran::evaluate::TypeParamInquiry &x);
  void show(const Fortran::evaluate::Triplet &x);
  void show(const Fortran::evaluate::Subscript &x);
  void show(const Fortran::evaluate::ArrayRef &x);
  void show(const Fortran::evaluate::CoarrayRef &x);
  void show(const Fortran::evaluate::DataRef &x);
  void show(const Fortran::evaluate::Substring &x);
  void show(const Fortran::evaluate::ComplexPart &x);
  template <typename T>
  void show(const Fortran::evaluate::Designator<T> &x) {
    indent("designator");
    show(x.u);
    outdent();
  }
  template <typename T>
  void show(const Fortran::evaluate::Variable<T> &x) {
    indent("variable");
    show(x.u);
    outdent();
  }
  void show(const Fortran::evaluate::DescriptorInquiry &x);
  void show(const Fortran::evaluate::SpecificIntrinsic &);
  void show(const Fortran::evaluate::ProcedureDesignator &x);
  void show(const Fortran::evaluate::ActualArgument &x);
  void show(const Fortran::evaluate::ProcedureRef &x) {
    indent("procedure ref");
    show(x.proc());
    show(x.arguments());
    outdent();
  }
  template <typename T>
  void show(const Fortran::evaluate::FunctionRef<T> &x) {
    indent("function ref");
    show(x.proc());
    show(x.arguments());
    outdent();
  }
  template <typename T>
  void show(const Fortran::evaluate::ArrayConstructorValue<T> &x) {
    show(x.u);
  }
  template <typename T>
  void show(const Fortran::evaluate::ArrayConstructorValues<T> &x) {
    indent("array constructor value");
    for (auto &v : x)
      show(v);
    outdent();
  }
  template <typename T>
  void show(const Fortran::evaluate::ImpliedDo<T> &x) {
    indent("implied do");
    show(x.lower());
    show(x.upper());
    show(x.stride());
    show(x.values());
    outdent();
  }
  void show(const Fortran::semantics::ParamValue &x);
  void
  show(const Fortran::semantics::DerivedTypeSpec::ParameterMapType::value_type
           &x);
  void show(const Fortran::semantics::DerivedTypeSpec &x);
  void show(const Fortran::evaluate::StructureConstructorValues::value_type &x);
  void show(const Fortran::evaluate::StructureConstructor &x);
  template <typename D, typename R, typename O>
  void show(const Fortran::evaluate::Operation<D, R, O> &op) {
    indent("unary op");
    show(op.left());
    outdent();
  }
  template <typename D, typename R, typename LO, typename RO>
  void show(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    indent("binary op");
    show(op.left());
    show(op.right());
    outdent();
  }
  void
  show(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x);
  template <typename T>
  void show(const Fortran::evaluate::Expr<T> &x) {
    indent("expr T");
    show(x.u);
    outdent();
  }

  const char *getIndentString() const;
  void print(llvm::Twine s);
  void indent(llvm::StringRef s);
  void outdent();

  llvm::raw_ostream &outs;
  unsigned level = 0;
};

LLVM_DUMP_METHOD void
dumpEvExpr(const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &x);
LLVM_DUMP_METHOD void dumpEvExpr(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 4>> &x);
LLVM_DUMP_METHOD void dumpEvExpr(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>> &x);
LLVM_DUMP_METHOD void dumpEvExpr(const Fortran::evaluate::ArrayRef &x);
LLVM_DUMP_METHOD void dumpEvExpr(const Fortran::evaluate::DataRef &x);
LLVM_DUMP_METHOD void dumpEvExpr(const Fortran::evaluate::Substring &x);
LLVM_DUMP_METHOD void dumpEvExpr(
    const Fortran::evaluate::Designator<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 4>> &x);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_DUMPEVALUATEEXPR_H
