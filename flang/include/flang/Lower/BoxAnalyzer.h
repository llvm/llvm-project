//===-- BoxAnalyzer.h -------------------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_BOXANALYZER_H
#define FORTRAN_LOWER_BOXANALYZER_H

#include "flang/Evaluate/fold.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/Matcher.h"

namespace Fortran::lower {

//===----------------------------------------------------------------------===//
// Classifications of a symbol.
//
// Each classification is a distinct class and can be used in pattern matching.
//===----------------------------------------------------------------------===//

namespace details {

using FromBox = std::monostate;

/// Base class for all box analysis results.
struct ScalarSym {
  ScalarSym(const Fortran::semantics::Symbol &sym) : sym{&sym} {}
  ScalarSym &operator=(const ScalarSym &) = default;

  const Fortran::semantics::Symbol &symbol() const { return *sym; }

  static constexpr bool staticSize() { return true; }
  static constexpr bool isChar() { return false; }
  static constexpr bool isArray() { return false; }

private:
  const Fortran::semantics::Symbol *sym;
};

/// Scalar of dependent type CHARACTER, constant LEN.
struct ScalarStaticChar : ScalarSym {
  ScalarStaticChar(const Fortran::semantics::Symbol &sym, int64_t len)
      : ScalarSym{sym}, len{len} {}

  int64_t charLen() const { return len; }

  static constexpr bool isChar() { return true; }

private:
  int64_t len;
};

/// Scalar of dependent type Derived, constant LEN(s).
struct ScalarStaticDerived : ScalarSym {
  ScalarStaticDerived(const Fortran::semantics::Symbol &sym,
                      llvm::SmallVectorImpl<int64_t> &&lens)
      : ScalarSym{sym}, lens{std::move(lens)} {}

private:
  llvm::SmallVector<int64_t> lens;
};

/// Scalar of dependent type CHARACTER, dynamic LEN.
struct ScalarDynamicChar : ScalarSym {
  ScalarDynamicChar(const Fortran::semantics::Symbol &sym,
                    const Fortran::lower::SomeExpr &len)
      : ScalarSym{sym}, len{len} {}
  ScalarDynamicChar(const Fortran::semantics::Symbol &sym)
      : ScalarSym{sym}, len{FromBox{}} {}

  llvm::Optional<Fortran::lower::SomeExpr> charLen() const {
    if (auto *l = std::get_if<Fortran::lower::SomeExpr>(&len))
      return {*l};
    return std::nullopt;
  }

  static constexpr bool staticSize() { return false; }
  static constexpr bool isChar() { return true; }

private:
  std::variant<FromBox, Fortran::lower::SomeExpr> len;
};

/// Scalar of dependent type Derived, dynamic LEN(s).
struct ScalarDynamicDerived : ScalarSym {
  ScalarDynamicDerived(const Fortran::semantics::Symbol &sym,
                       llvm::SmallVectorImpl<Fortran::lower::SomeExpr> &&lens)
      : ScalarSym{sym}, lens{std::move(lens)} {}

private:
  llvm::SmallVector<Fortran::lower::SomeExpr> lens;
};

struct LBoundsAndShape {
  LBoundsAndShape(llvm::SmallVectorImpl<int64_t> &&lbounds,
                  llvm::SmallVectorImpl<int64_t> &&shapes)
      : lbounds{std::move(lbounds)}, shapes{std::move(shapes)} {}

  static constexpr bool staticSize() { return true; }
  static constexpr bool isArray() { return true; }
  bool lboundAllOnes() const {
    return llvm::all_of(lbounds, [](int64_t v) { return v == 1; });
  }

  llvm::SmallVector<int64_t> lbounds;
  llvm::SmallVector<int64_t> shapes;
};

/// Array of T with statically known origin (lbounds) and shape.
struct StaticArray : ScalarSym, LBoundsAndShape {
  StaticArray(const Fortran::semantics::Symbol &sym,
              llvm::SmallVectorImpl<int64_t> &&lbounds,
              llvm::SmallVectorImpl<int64_t> &&shapes)
      : ScalarSym{sym}, LBoundsAndShape{std::move(lbounds), std::move(shapes)} {
  }

  static constexpr bool staticSize() { return LBoundsAndShape::staticSize(); }
};

struct DynamicBound {
  DynamicBound(
      llvm::SmallVectorImpl<const Fortran::semantics::ShapeSpec *> &&bounds)
      : bounds{std::move(bounds)} {}

  static constexpr bool staticSize() { return false; }
  static constexpr bool isArray() { return true; }
  bool lboundAllOnes() const {
    return llvm::all_of(bounds, [](const Fortran::semantics::ShapeSpec *p) {
      if (auto low = p->lbound().GetExplicit())
        if (auto lb = Fortran::evaluate::ToInt64(*low))
          return *lb == 1;
      return false;
    });
  }

  llvm::SmallVector<const Fortran::semantics::ShapeSpec *> bounds;
};

/// Array of T with dynamic origin and/or shape.
struct DynamicArray : ScalarSym, DynamicBound {
  DynamicArray(
      const Fortran::semantics::Symbol &sym,
      llvm::SmallVectorImpl<const Fortran::semantics::ShapeSpec *> &&bounds)
      : ScalarSym{sym}, DynamicBound{std::move(bounds)} {}

  static constexpr bool staticSize() { return DynamicBound::staticSize(); }
};

/// Array of CHARACTER with statically known LEN, origin, and shape.
struct StaticArrayStaticChar : ScalarStaticChar, LBoundsAndShape {
  StaticArrayStaticChar(const Fortran::semantics::Symbol &sym, int64_t len,
                        llvm::SmallVectorImpl<int64_t> &&lbounds,
                        llvm::SmallVectorImpl<int64_t> &&shapes)
      : ScalarStaticChar{sym, len}, LBoundsAndShape{std::move(lbounds),
                                                    std::move(shapes)} {}

  static constexpr bool staticSize() {
    return ScalarStaticChar::staticSize() && LBoundsAndShape::staticSize();
  }
};

/// Array of CHARACTER with dynamic LEN but constant origin, shape.
struct StaticArrayDynamicChar : ScalarDynamicChar, LBoundsAndShape {
  StaticArrayDynamicChar(const Fortran::semantics::Symbol &sym,
                         const Fortran::lower::SomeExpr &len,
                         llvm::SmallVectorImpl<int64_t> &&lbounds,
                         llvm::SmallVectorImpl<int64_t> &&shapes)
      : ScalarDynamicChar{sym, len}, LBoundsAndShape{std::move(lbounds),
                                                     std::move(shapes)} {}
  StaticArrayDynamicChar(const Fortran::semantics::Symbol &sym,
                         llvm::SmallVectorImpl<int64_t> &&lbounds,
                         llvm::SmallVectorImpl<int64_t> &&shapes)
      : ScalarDynamicChar{sym}, LBoundsAndShape{std::move(lbounds),
                                                std::move(shapes)} {}

  static constexpr bool staticSize() {
    return ScalarDynamicChar::staticSize() && LBoundsAndShape::staticSize();
  }
};

/// Array of CHARACTER with constant LEN but dynamic origin, shape.
struct DynamicArrayStaticChar : ScalarStaticChar, DynamicBound {
  DynamicArrayStaticChar(
      const Fortran::semantics::Symbol &sym, int64_t len,
      llvm::SmallVectorImpl<const Fortran::semantics::ShapeSpec *> &&bounds)
      : ScalarStaticChar{sym, len}, DynamicBound{std::move(bounds)} {}

  static constexpr bool staticSize() {
    return ScalarStaticChar::staticSize() && DynamicBound::staticSize();
  }
};

/// Array of CHARACTER with dynamic LEN, origin, and shape.
struct DynamicArrayDynamicChar : ScalarDynamicChar, DynamicBound {
  DynamicArrayDynamicChar(
      const Fortran::semantics::Symbol &sym,
      const Fortran::lower::SomeExpr &len,
      llvm::SmallVectorImpl<const Fortran::semantics::ShapeSpec *> &&bounds)
      : ScalarDynamicChar{sym, len}, DynamicBound{std::move(bounds)} {}
  DynamicArrayDynamicChar(
      const Fortran::semantics::Symbol &sym,
      llvm::SmallVectorImpl<const Fortran::semantics::ShapeSpec *> &&bounds)
      : ScalarDynamicChar{sym}, DynamicBound{std::move(bounds)} {}

  static constexpr bool staticSize() {
    return ScalarDynamicChar::staticSize() && DynamicBound::staticSize();
  }
};

// TODO: Arrays of derived types with LEN(s)...

} // namespace details

inline bool symIsChar(const Fortran::semantics::Symbol &sym) {
  return sym.GetType()->category() ==
         Fortran::semantics::DeclTypeSpec::Character;
}

inline bool symIsArray(const Fortran::semantics::Symbol &sym) {
  const auto *det =
      sym.GetUltimate().detailsIf<Fortran::semantics::ObjectEntityDetails>();
  return det && det->IsArray();
}

inline bool isExplicitShape(const Fortran::semantics::Symbol &sym) {
  const auto *det =
      sym.GetUltimate().detailsIf<Fortran::semantics::ObjectEntityDetails>();
  return det && det->IsArray() && det->shape().IsExplicitShape();
}

inline bool isAssumedSize(const Fortran::semantics::Symbol &sym) {
  return Fortran::semantics::IsAssumedSizeArray(sym.GetUltimate());
}

//===----------------------------------------------------------------------===//
// Perform analysis to determine a box's parameter values
//===----------------------------------------------------------------------===//

/// Analyze a symbol, classify it as to whether it just a scalar, a CHARACTER
/// scalar, an array entity, a combination thereof, and whether the LEN, shape,
/// and lbounds are constant or not.
class BoxAnalyzer : public fir::details::matcher<BoxAnalyzer> {
public:
  // Analysis default state
  using None = std::monostate;

  using ScalarSym = details::ScalarSym;
  using ScalarStaticChar = details::ScalarStaticChar;
  using ScalarDynamicChar = details::ScalarDynamicChar;
  using StaticArray = details::StaticArray;
  using DynamicArray = details::DynamicArray;
  using StaticArrayStaticChar = details::StaticArrayStaticChar;
  using StaticArrayDynamicChar = details::StaticArrayDynamicChar;
  using DynamicArrayStaticChar = details::DynamicArrayStaticChar;
  using DynamicArrayDynamicChar = details::DynamicArrayDynamicChar;
  // TODO: derived types

  using VT = std::variant<None, ScalarSym, ScalarStaticChar, ScalarDynamicChar,
                          StaticArray, DynamicArray, StaticArrayStaticChar,
                          StaticArrayDynamicChar, DynamicArrayStaticChar,
                          DynamicArrayDynamicChar>;

  //===--------------------------------------------------------------------===//
  // Constructor
  //===--------------------------------------------------------------------===//

  BoxAnalyzer() : box{None{}} {}

  operator bool() const { return !std::holds_alternative<None>(box); }

  bool isTrivial() const { return std::holds_alternative<ScalarSym>(box); }

  /// Returns true for any sort of CHARACTER.
  bool isChar() const {
    return match([](const ScalarStaticChar &) { return true; },
                 [](const ScalarDynamicChar &) { return true; },
                 [](const StaticArrayStaticChar &) { return true; },
                 [](const StaticArrayDynamicChar &) { return true; },
                 [](const DynamicArrayStaticChar &) { return true; },
                 [](const DynamicArrayDynamicChar &) { return true; },
                 [](const auto &) { return false; });
  }

  /// Returns true for any sort of array.
  bool isArray() const {
    return match([](const StaticArray &) { return true; },
                 [](const DynamicArray &) { return true; },
                 [](const StaticArrayStaticChar &) { return true; },
                 [](const StaticArrayDynamicChar &) { return true; },
                 [](const DynamicArrayStaticChar &) { return true; },
                 [](const DynamicArrayDynamicChar &) { return true; },
                 [](const auto &) { return false; });
  }

  /// Returns true iff this is an array with constant extents and lbounds. This
  /// returns true for arrays of CHARACTER, even if the LEN is not a constant.
  bool isStaticArray() const {
    return match([](const StaticArray &) { return true; },
                 [](const StaticArrayStaticChar &) { return true; },
                 [](const StaticArrayDynamicChar &) { return true; },
                 [](const auto &) { return false; });
  }

  bool isConstant() const {
    return match(
        [](const None &) -> bool {
          llvm::report_fatal_error("internal: analysis failed");
        },
        [](const auto &x) { return x.staticSize(); });
  }

  llvm::Optional<int64_t> getCharLenConst() const {
    using A = llvm::Optional<int64_t>;
    return match(
        [](const ScalarStaticChar &x) -> A { return {x.charLen()}; },
        [](const StaticArrayStaticChar &x) -> A { return {x.charLen()}; },
        [](const DynamicArrayStaticChar &x) -> A { return {x.charLen()}; },
        [](const auto &) -> A { return std::nullopt; });
  }

  llvm::Optional<Fortran::lower::SomeExpr> getCharLenExpr() const {
    using A = llvm::Optional<Fortran::lower::SomeExpr>;
    return match([](const ScalarDynamicChar &x) { return x.charLen(); },
                 [](const StaticArrayDynamicChar &x) { return x.charLen(); },
                 [](const DynamicArrayDynamicChar &x) { return x.charLen(); },
                 [](const auto &) -> A { return std::nullopt; });
  }

  /// Is the origin of this array the default of vector of `1`?
  bool lboundIsAllOnes() const {
    return match(
        [&](const StaticArray &x) { return x.lboundAllOnes(); },
        [&](const DynamicArray &x) { return x.lboundAllOnes(); },
        [&](const StaticArrayStaticChar &x) { return x.lboundAllOnes(); },
        [&](const StaticArrayDynamicChar &x) { return x.lboundAllOnes(); },
        [&](const DynamicArrayStaticChar &x) { return x.lboundAllOnes(); },
        [&](const DynamicArrayDynamicChar &x) { return x.lboundAllOnes(); },
        [](const auto &) -> bool { llvm::report_fatal_error("not an array"); });
  }

  /// Get the static lbound values (the origin of the array).
  llvm::ArrayRef<int64_t> staticLBound() const {
    using A = llvm::ArrayRef<int64_t>;
    return match([](const StaticArray &x) -> A { return x.lbounds; },
                 [](const StaticArrayStaticChar &x) -> A { return x.lbounds; },
                 [](const StaticArrayDynamicChar &x) -> A { return x.lbounds; },
                 [](const auto &) -> A {
                   llvm::report_fatal_error("does not have static lbounds");
                 });
  }

  /// Get the static extents of the array.
  llvm::ArrayRef<int64_t> staticShape() const {
    using A = llvm::ArrayRef<int64_t>;
    return match([](const StaticArray &x) -> A { return x.shapes; },
                 [](const StaticArrayStaticChar &x) -> A { return x.shapes; },
                 [](const StaticArrayDynamicChar &x) -> A { return x.shapes; },
                 [](const auto &) -> A {
                   llvm::report_fatal_error("does not have static shape");
                 });
  }

  /// Get the dynamic bounds information of the array (both origin, shape).
  llvm::ArrayRef<const Fortran::semantics::ShapeSpec *> dynamicBound() const {
    using A = llvm::ArrayRef<const Fortran::semantics::ShapeSpec *>;
    return match([](const DynamicArray &x) -> A { return x.bounds; },
                 [](const DynamicArrayStaticChar &x) -> A { return x.bounds; },
                 [](const DynamicArrayDynamicChar &x) -> A { return x.bounds; },
                 [](const auto &) -> A {
                   llvm::report_fatal_error("does not have bounds");
                 });
  }

  /// Run the analysis on `sym`.
  void analyze(const Fortran::semantics::Symbol &sym) {
    if (symIsArray(sym)) {
      bool isConstant = !isAssumedSize(sym);
      llvm::SmallVector<int64_t> lbounds;
      llvm::SmallVector<int64_t> shapes;
      llvm::SmallVector<const Fortran::semantics::ShapeSpec *> bounds;
      for (const Fortran::semantics::ShapeSpec &subs : getSymShape(sym)) {
        bounds.push_back(&subs);
        if (!isConstant)
          continue;
        if (auto low = subs.lbound().GetExplicit()) {
          if (auto lb = Fortran::evaluate::ToInt64(*low)) {
            lbounds.push_back(*lb); // origin for this dim
            if (auto high = subs.ubound().GetExplicit()) {
              if (auto ub = Fortran::evaluate::ToInt64(*high)) {
                int64_t extent = *ub - *lb + 1;
                shapes.push_back(extent < 0 ? 0 : extent);
                continue;
              }
            } else if (subs.ubound().isStar()) {
              assert(Fortran::semantics::IsNamedConstant(sym) &&
                     "expect implied shape constant");
              shapes.push_back(fir::SequenceType::getUnknownExtent());
              continue;
            }
          }
        }
        isConstant = false;
      }

      // sym : array<CHARACTER>
      if (symIsChar(sym)) {
        if (auto len = charLenConstant(sym)) {
          if (isConstant)
            box = StaticArrayStaticChar(sym, *len, std::move(lbounds),
                                        std::move(shapes));
          else
            box = DynamicArrayStaticChar(sym, *len, std::move(bounds));
          return;
        }
        if (auto var = charLenVariable(sym)) {
          if (isConstant)
            box = StaticArrayDynamicChar(sym, *var, std::move(lbounds),
                                         std::move(shapes));
          else
            box = DynamicArrayDynamicChar(sym, *var, std::move(bounds));
          return;
        }
        if (isConstant)
          box = StaticArrayDynamicChar(sym, std::move(lbounds),
                                       std::move(shapes));
        else
          box = DynamicArrayDynamicChar(sym, std::move(bounds));
        return;
      }

      // sym : array<other>
      if (isConstant)
        box = StaticArray(sym, std::move(lbounds), std::move(shapes));
      else
        box = DynamicArray(sym, std::move(bounds));
      return;
    }

    // sym : CHARACTER
    if (symIsChar(sym)) {
      if (auto len = charLenConstant(sym))
        box = ScalarStaticChar(sym, *len);
      else if (auto var = charLenVariable(sym))
        box = ScalarDynamicChar(sym, *var);
      else
        box = ScalarDynamicChar(sym);
      return;
    }

    // sym : other
    box = ScalarSym(sym);
  }

  const VT &matchee() const { return box; }

private:
  // Get the shape of a symbol.
  const Fortran::semantics::ArraySpec &
  getSymShape(const Fortran::semantics::Symbol &sym) {
    return sym.GetUltimate()
        .get<Fortran::semantics::ObjectEntityDetails>()
        .shape();
  }

  // Get the constant LEN of a CHARACTER, if it exists.
  llvm::Optional<int64_t>
  charLenConstant(const Fortran::semantics::Symbol &sym) {
    if (llvm::Optional<Fortran::lower::SomeExpr> expr = charLenVariable(sym))
      if (std::optional<int64_t> asInt = Fortran::evaluate::ToInt64(*expr)) {
        // Length is max(0, *asInt) (F2018 7.4.4.2 point 5.).
        if (*asInt < 0)
          return 0;
        return *asInt;
      }
    return std::nullopt;
  }

  // Get the `SomeExpr` that describes the CHARACTER's LEN.
  llvm::Optional<Fortran::lower::SomeExpr>
  charLenVariable(const Fortran::semantics::Symbol &sym) {
    const Fortran::semantics::ParamValue &lenParam =
        sym.GetType()->characterTypeSpec().length();
    if (Fortran::semantics::MaybeIntExpr expr = lenParam.GetExplicit())
      return {Fortran::evaluate::AsGenericExpr(std::move(*expr))};
    // For assumed LEN parameters, the length comes from the initialization
    // expression.
    if (sym.attrs().test(Fortran::semantics::Attr::PARAMETER))
      if (const auto *objectDetails =
              sym.GetUltimate()
                  .detailsIf<Fortran::semantics::ObjectEntityDetails>())
        if (objectDetails->init())
          if (const auto *charExpr = std::get_if<
                  Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(
                  &objectDetails->init()->u))
            if (Fortran::semantics::MaybeSubscriptIntExpr expr =
                    charExpr->LEN())
              return {Fortran::evaluate::AsGenericExpr(std::move(*expr))};
    return std::nullopt;
  }

  VT box;
}; // namespace Fortran::lower

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_BOXANALYZER_H
