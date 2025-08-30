//===-- Semantics/dump-expr.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_DUMPEXPR_H
#define FORTRAN_SEMANTICS_DUMPEXPR_H

#include "flang/Evaluate/tools.h"
#include "flang/Semantics/symbol.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Fortran::semantics {

/// Class to dump evaluate::Expr trees out in a user readable way.
///
/// FIXME: This can be improved to dump more information in some cases.
class DumpEvaluateExpr {
public:
  DumpEvaluateExpr() : outs_(llvm::errs()) {}
  DumpEvaluateExpr(llvm::raw_ostream &str) : outs_(str) {}

  template <typename A> static void Dump(const A &x) {
    DumpEvaluateExpr{}.Show(x);
  }
  template <typename A>
  static void Dump(llvm::raw_ostream &stream, const A &x) {
    DumpEvaluateExpr{stream}.Show(x);
  }

private:
  template <typename T> struct TypeOf {
    static constexpr std::string_view get() {
#if defined(__GNUC__)
#define DUMP_EXPR_SHOW_TYPE
      std::string_view v(__PRETTY_FUNCTION__);
      // Extract the "xyz" from the "pretty function" string:
      // "... [with T = xyz; std::string_view = ...]"
      std::string_view front("with T = ");
      std::string_view back("; std::string_view =");

#elif defined(_MSC_VER)
#define DUMP_EXPR_SHOW_TYPE
      std::string_view v(__FUNCSIG__);
      // Extract the "xyz" from the "pretty function" string:
      // "...TypeOf<xyz>::get(void)"
      std::string_view front("TypeOf<");
      std::string_view back(">::get(void)");

#endif

#if defined(DUMP_EXPR_SHOW_TYPE)
#undef DUMP_EXPR_SHOW_TYPE
      if (auto fpos{v.find(front)}; fpos != v.npos) {
        v.remove_prefix(fpos + front.size());
        if (auto bpos{v.find(back)}; bpos != v.npos) {
          v.remove_suffix(v.size() - bpos);
          return v;
        }
      }
#endif

      return "";
    }

    static constexpr std::string_view name{TypeOf<T>::get()};
  };

  template <typename A, bool C> void Show(const common::Indirection<A, C> &x) {
    Show(x.value());
  }
  template <typename A> void Show(const SymbolRef x) { Show(*x); }
  template <typename A> void Show(const std::unique_ptr<A> &x) {
    Show(x.get());
  }
  template <typename A> void Show(const std::shared_ptr<A> &x) {
    Show(x.get());
  }
  template <typename A> void Show(const A *x) {
    if (x) {
      Show(*x);
      return;
    }
    Print("nullptr");
  }
  template <typename A> void Show(const std::optional<A> &x) {
    if (x) {
      Show(*x);
      return;
    }
    Print("None");
  }
  template <typename... A> void Show(const std::variant<A...> &u) {
    common::visit([&](const auto &v) { Show(v); }, u);
  }
  template <typename A> void Show(const std::vector<A> &x) {
    Indent("vector");
    for (const auto &v : x) {
      Show(v);
    }
    Outdent();
  }
  void Show(const evaluate::BOZLiteralConstant &);
  void Show(const evaluate::NullPointer &);
  template <typename T> void Show(const evaluate::Constant<T> &x) {
    if constexpr (T::category == common::TypeCategory::Derived) {
      Indent("derived constant "s + std::string(TypeOf<T>::name));
      for (const auto &map : x.values()) {
        for (const auto &pair : map) {
          Show(pair.second.value());
        }
      }
      Outdent();
    } else {
      Print("constant "s + std::string(TypeOf<T>::name));
    }
  }
  void Show(const Symbol &symbol);
  void Show(const evaluate::StaticDataObject &);
  void Show(const evaluate::ImpliedDoIndex &);
  void Show(const evaluate::BaseObject &x);
  void Show(const evaluate::Component &x);
  void Show(const evaluate::NamedEntity &x);
  void Show(const evaluate::TypeParamInquiry &x);
  void Show(const evaluate::Triplet &x);
  void Show(const evaluate::Subscript &x);
  void Show(const evaluate::ArrayRef &x);
  void Show(const evaluate::CoarrayRef &x);
  void Show(const evaluate::DataRef &x);
  void Show(const evaluate::Substring &x);
  void Show(const evaluate::ComplexPart &x);
  template <typename T> void Show(const evaluate::Designator<T> &x) {
    Indent("designator "s + std::string(TypeOf<T>::name));
    Show(x.u);
    Outdent();
  }
  void Show(const evaluate::DescriptorInquiry &x);
  void Show(const evaluate::SpecificIntrinsic &);
  void Show(const evaluate::ProcedureDesignator &x);
  void Show(const evaluate::ActualArgument &x);
  void Show(const evaluate::ProcedureRef &x) {
    Indent("procedure ref");
    Show(x.proc());
    Show(x.arguments());
    Outdent();
  }
  template <typename T> void Show(const evaluate::FunctionRef<T> &x) {
    Indent("function ref "s + std::string(TypeOf<T>::name));
    Show(x.proc());
    Show(x.arguments());
    Outdent();
  }
  template <typename T> void Show(const evaluate::ArrayConstructorValue<T> &x) {
    Show(x.u);
  }
  template <typename T>
  void Show(const evaluate::ArrayConstructorValues<T> &x) {
    Indent("array constructor value "s + std::string(TypeOf<T>::name));
    for (auto &v : x) {
      Show(v);
    }
    Outdent();
  }
  template <typename T> void Show(const evaluate::ImpliedDo<T> &x) {
    Indent("implied do "s + std::string(TypeOf<T>::name));
    Show(x.lower());
    Show(x.upper());
    Show(x.stride());
    Show(x.values());
    Outdent();
  }
  void Show(const ParamValue &x);
  void Show(const DerivedTypeSpec::ParameterMapType::value_type &x);
  void Show(const DerivedTypeSpec &x);
  void Show(const evaluate::StructureConstructorValues::value_type &x);
  void Show(const evaluate::StructureConstructor &x);
  template <typename D, typename R, typename O>
  void Show(const evaluate::Operation<D, R, O> &op) {
    Indent("unary op "s + std::string(TypeOf<D>::name));
    Show(op.left());
    Outdent();
  }
  template <typename D, typename R, typename LO, typename RO>
  void Show(const evaluate::Operation<D, R, LO, RO> &op) {
    Indent("binary op "s + std::string(TypeOf<D>::name));
    Show(op.left());
    Show(op.right());
    Outdent();
  }
  void Show(const evaluate::Relational<evaluate::SomeType> &x);
  template <typename T> void Show(const evaluate::Expr<T> &x) {
    Indent("expr <" + std::string(TypeOf<T>::name) + ">");
    Show(x.u);
    Outdent();
  }

  const char *GetIndentString() const;
  void Print(llvm::Twine s);
  void Indent(llvm::StringRef s);
  void Outdent();

  llvm::raw_ostream &outs_;
  unsigned level_{0};
};

LLVM_DUMP_METHOD void DumpEvExpr(const evaluate::Expr<evaluate::SomeType> &x);
LLVM_DUMP_METHOD void DumpEvExpr(
    const evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 4>> &x);
LLVM_DUMP_METHOD void DumpEvExpr(
    const evaluate::Expr<evaluate::Type<common::TypeCategory::Integer, 8>> &x);
LLVM_DUMP_METHOD void DumpEvExpr(const evaluate::ArrayRef &x);
LLVM_DUMP_METHOD void DumpEvExpr(const evaluate::DataRef &x);
LLVM_DUMP_METHOD void DumpEvExpr(const evaluate::Substring &x);
LLVM_DUMP_METHOD void DumpEvExpr(
    const evaluate::Designator<evaluate::Type<common::TypeCategory::Integer, 4>>
        &x);

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_DUMPEXPR_H
