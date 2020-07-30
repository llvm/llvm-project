//===-- Optimizer/Dialect/FIRAttr.h -- FIR attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DIALECT_FIRATTR_H
#define OPTIMIZER_DIALECT_FIRATTR_H

#include "mlir/IR/Attributes.h"

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir

namespace fir {

class FIROpsDialect;

namespace detail {
struct RealAttributeStorage;
struct TypeAttributeStorage;
} // namespace detail

enum AttributeKind {
  FIR_ATTR = mlir::Attribute::FIRST_FIR_ATTR,
  FIR_EXACTTYPE, // instance_of, precise type relation
  FIR_SUBCLASS,  // subsumed_by, is-a (subclass) relation
  FIR_POINT,
  FIR_CLOSEDCLOSED_INTERVAL,
  FIR_OPENCLOSED_INTERVAL,
  FIR_CLOSEDOPEN_INTERVAL,
  FIR_REAL_ATTR,
};

class ExactTypeAttr
    : public mlir::Attribute::AttrBase<ExactTypeAttr, mlir::Attribute,
                                       detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = mlir::Type;

  static constexpr llvm::StringRef getAttrName() { return "instance"; }
  static ExactTypeAttr get(mlir::Type value);

  mlir::Type getType() const;

  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() { return AttributeKind::FIR_EXACTTYPE; }
};

class SubclassAttr
    : public mlir::Attribute::AttrBase<SubclassAttr, mlir::Attribute,
                                       detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = mlir::Type;

  static constexpr llvm::StringRef getAttrName() { return "subsumed"; }
  static SubclassAttr get(mlir::Type value);

  mlir::Type getType() const;

  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() { return AttributeKind::FIR_SUBCLASS; }
};

// Attributes for building SELECT CASE multiway branches

/// A closed interval (including the bound values) is an interval with both an
/// upper and lower bound as given as ssa-values.
/// A case selector of `CASE (n:m)` corresponds to any value from `n` to `m` and
/// is encoded as `#fir.interval, %n, %m`.
class ClosedIntervalAttr
    : public mlir::Attribute::AttrBase<ClosedIntervalAttr, mlir::Attribute,
                                       mlir::AttributeStorage> {
public:
  using Base::Base;

  static constexpr llvm::StringRef getAttrName() { return "interval"; }
  static ClosedIntervalAttr get(mlir::MLIRContext *ctxt);
  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() {
    return AttributeKind::FIR_CLOSEDCLOSED_INTERVAL;
  }
};

/// An upper bound is an open interval (including the bound value) as given as
/// an ssa-value.
/// A case selector of `CASE (:m)` corresponds to any value up to and including
/// `m` and is encoded as `#fir.upper, %m`.
class UpperBoundAttr
    : public mlir::Attribute::AttrBase<UpperBoundAttr, mlir::Attribute,
                                       mlir::AttributeStorage> {
public:
  using Base::Base;

  static constexpr llvm::StringRef getAttrName() { return "upper"; }
  static UpperBoundAttr get(mlir::MLIRContext *ctxt);
  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() {
    return AttributeKind::FIR_OPENCLOSED_INTERVAL;
  }
};

/// A lower bound is an open interval (including the bound value) as given as
/// an ssa-value.
/// A case selector of `CASE (n:)` corresponds to any value down to and
/// including `n` and is encoded as `#fir.lower, %n`.
class LowerBoundAttr
    : public mlir::Attribute::AttrBase<LowerBoundAttr, mlir::Attribute,
                                       mlir::AttributeStorage> {
public:
  using Base::Base;

  static constexpr llvm::StringRef getAttrName() { return "lower"; }
  static LowerBoundAttr get(mlir::MLIRContext *ctxt);
  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() {
    return AttributeKind::FIR_CLOSEDOPEN_INTERVAL;
  }
};

/// A pointer interval is a closed interval as given as an ssa-value. The
/// interval contains exactly one value.
/// A case selector of `CASE (p)` corresponds to exactly the value `p` and is
/// encoded as `#fir.point, %p`.
class PointIntervalAttr
    : public mlir::Attribute::AttrBase<PointIntervalAttr, mlir::Attribute,
                                       mlir::AttributeStorage> {
public:
  using Base::Base;

  static constexpr llvm::StringRef getAttrName() { return "point"; }
  static PointIntervalAttr get(mlir::MLIRContext *ctxt);
  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() { return AttributeKind::FIR_POINT; }
};

/// A real attribute is used to workaround MLIR's default parsing of a real
/// constant.
/// `#fir.real<10, 3.14>` is used to introduce a real constant of value `3.14`
/// with a kind of `10`.
class RealAttr
    : public mlir::Attribute::AttrBase<RealAttr, mlir::Attribute,
                                       detail::RealAttributeStorage> {
public:
  using Base::Base;
  using ValueType = std::pair<int, llvm::APFloat>;

  static constexpr llvm::StringRef getAttrName() { return "real"; }
  static RealAttr get(mlir::MLIRContext *ctxt, const ValueType &key);

  int getFKind() const;
  llvm::APFloat getValue() const;

  static constexpr bool kindof(unsigned kind) { return kind == getId(); }
  static constexpr unsigned getId() { return AttributeKind::FIR_REAL_ATTR; }
};

mlir::Attribute parseFirAttribute(FIROpsDialect *dialect,
                                  mlir::DialectAsmParser &parser,
                                  mlir::Type type);

void printFirAttribute(FIROpsDialect *dialect, mlir::Attribute attr,
                       mlir::DialectAsmPrinter &p);

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIRATTR_H
