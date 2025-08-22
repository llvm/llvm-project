//===-- Lower/Support/Utils.h -- utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_SUPPORT_UTILS_H
#define FORTRAN_LOWER_SUPPORT_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

namespace Fortran::lower {
using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
} // end namespace Fortran::lower

//===----------------------------------------------------------------------===//
// Small inline helper functions to deal with repetitive, clumsy conversions.
//===----------------------------------------------------------------------===//

/// Convert an F18 CharBlock to an LLVM StringRef.
inline llvm::StringRef toStringRef(const Fortran::parser::CharBlock &cb) {
  return {cb.begin(), cb.size()};
}

/// Template helper to remove Fortran::common::Indirection wrappers.
template <typename A>
const A &removeIndirection(const A &a) {
  return a;
}
template <typename A>
const A &removeIndirection(const Fortran::common::Indirection<A> &a) {
  return a.value();
}

/// Clone subexpression and wrap it as a generic `Fortran::evaluate::Expr`.
template <typename A>
static Fortran::lower::SomeExpr toEvExpr(const A &x) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x));
}

template <Fortran::common::TypeCategory FROM>
static Fortran::lower::SomeExpr ignoreEvConvert(
    const Fortran::evaluate::Convert<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
        FROM> &x) {
  return toEvExpr(x.left());
}
template <typename A>
static Fortran::lower::SomeExpr ignoreEvConvert(const A &x) {
  return toEvExpr(x);
}

/// A vector subscript expression may be wrapped with a cast to INTEGER*8.
/// Get rid of it here so the vector can be loaded. Add it back when
/// generating the elemental evaluation (inside the loop nest).
inline Fortran::lower::SomeExpr
ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                    Fortran::common::TypeCategory::Integer, 8>> &x) {
  return Fortran::common::visit(
      [](const auto &v) { return ignoreEvConvert(v); }, x.u);
}

/// Zip two containers of the same size together and flatten the pairs. `flatZip
/// [1;2] [3;4]` yields `[1;3;2;4]`.
template <typename A>
A flatZip(const A &container1, const A &container2) {
  assert(container1.size() == container2.size());
  A result;
  for (auto [e1, e2] : llvm::zip(container1, container2)) {
    result.emplace_back(e1);
    result.emplace_back(e2);
  }
  return result;
}

namespace Fortran::lower {
unsigned getHashValue(const Fortran::lower::SomeExpr *x);
unsigned getHashValue(const Fortran::lower::ExplicitIterSpace::ArrayBases &x);

bool isEqual(const Fortran::lower::SomeExpr *x,
             const Fortran::lower::SomeExpr *y);
bool isEqual(const Fortran::lower::ExplicitIterSpace::ArrayBases &x,
             const Fortran::lower::ExplicitIterSpace::ArrayBases &y);

template <typename OpType, typename OperandsStructType>
void privatizeSymbol(
    lower::AbstractConverter &converter, fir::FirOpBuilder &firOpBuilder,
    lower::SymMap &symTable,
    llvm::SetVector<const semantics::Symbol *> &allPrivatizedSymbols,
    llvm::SmallPtrSet<const semantics::Symbol *, 16> &mightHaveReadHostSym,
    const semantics::Symbol *symToPrivatize, OperandsStructType *clauseOps);

} // end namespace Fortran::lower

// DenseMapInfo for pointers to Fortran::lower::SomeExpr.
namespace llvm {
template <>
struct DenseMapInfo<const Fortran::lower::SomeExpr *> {
  static inline const Fortran::lower::SomeExpr *getEmptyKey() {
    return reinterpret_cast<Fortran::lower::SomeExpr *>(~0);
  }
  static inline const Fortran::lower::SomeExpr *getTombstoneKey() {
    return reinterpret_cast<Fortran::lower::SomeExpr *>(~0 - 1);
  }
  static unsigned getHashValue(const Fortran::lower::SomeExpr *v) {
    return Fortran::lower::getHashValue(v);
  }
  static bool isEqual(const Fortran::lower::SomeExpr *lhs,
                      const Fortran::lower::SomeExpr *rhs) {
    return Fortran::lower::isEqual(lhs, rhs);
  }
};
} // namespace llvm

#endif // FORTRAN_LOWER_SUPPORT_UTILS_H
