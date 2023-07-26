//===-- include/flang/Evaluate/fold-designator.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_FOLD_DESIGNATOR_H_
#define FORTRAN_EVALUATE_FOLD_DESIGNATOR_H_

// Resolves a designator at compilation time to a base symbol, a byte offset
// from that symbol, and a byte size.  Also resolves in the reverse direction,
// reconstructing a designator from a symbol, byte offset, and size.
// Used for resolving variables in DATA statements to ranges in their
// initial images.
// Some designators can also be folded into constant pointer descriptors,
// which also have per-dimension extent and stride information suitable
// for initializing a descriptor.
// (The designators that cannot be folded are those with vector-valued
// subscripts; they are allowed as DATA statement objects, but are not valid
// initial pointer targets.)

#include "common.h"
#include "expression.h"
#include "fold.h"
#include "shape.h"
#include "type.h"
#include "variable.h"
#include <optional>
#include <variant>

namespace Fortran::evaluate {

using common::ConstantSubscript;

// Identifies a single contiguous interval of bytes at a fixed offset
// from a known symbol.
class OffsetSymbol {
public:
  OffsetSymbol(const Symbol &symbol, std::size_t bytes)
      : symbol_{symbol}, size_{bytes} {}
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(OffsetSymbol)

  const Symbol &symbol() const { return *symbol_; }
  void set_symbol(const Symbol &symbol) { symbol_ = symbol; };
  ConstantSubscript offset() const { return offset_; }
  void Augment(ConstantSubscript n) { offset_ += n; }
  std::size_t size() const { return size_; }
  void set_size(std::size_t bytes) { size_ = bytes; }

private:
  SymbolRef symbol_;
  ConstantSubscript offset_{0};
  std::size_t size_;
};

// Folds a Designator<T> into a sequence of OffsetSymbols, if it can
// be so folded.  Array sections yield multiple results, each
// corresponding to an element in array element order.
class DesignatorFolder {
public:
  explicit DesignatorFolder(FoldingContext &c, bool getLastComponent = false)
      : context_{c}, getLastComponent_{getLastComponent} {}

  bool isEmpty() const { return isEmpty_; }
  bool isOutOfRange() const { return isOutOfRange_; }

  template <typename T>
  std::optional<OffsetSymbol> FoldDesignator(const Expr<T> &expr) {
    return common::visit(
        [&](const auto &x) { return FoldDesignator(x, elementNumber_++); },
        expr.u);
  }

private:
  std::optional<OffsetSymbol> FoldDesignator(const Symbol &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const SymbolRef &x, ConstantSubscript which) {
    return FoldDesignator(*x, which);
  }
  std::optional<OffsetSymbol> FoldDesignator(
      const ArrayRef &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const Component &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const ComplexPart &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const Substring &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const DataRef &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const NamedEntity &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const CoarrayRef &, ConstantSubscript);
  std::optional<OffsetSymbol> FoldDesignator(
      const ProcedureDesignator &, ConstantSubscript);

  template <typename T>
  std::optional<OffsetSymbol> FoldDesignator(
      const Expr<T> &expr, ConstantSubscript which) {
    return common::visit(
        [&](const auto &x) { return FoldDesignator(x, which); }, expr.u);
  }

  template <typename A>
  std::optional<OffsetSymbol> FoldDesignator(const A &, ConstantSubscript) {
    return std::nullopt;
  }

  template <typename T>
  std::optional<OffsetSymbol> FoldDesignator(
      const Designator<T> &designator, ConstantSubscript which) {
    return common::visit(
        [&](const auto &x) { return FoldDesignator(x, which); }, designator.u);
  }
  template <int KIND>
  std::optional<OffsetSymbol> FoldDesignator(
      const Designator<Type<TypeCategory::Character, KIND>> &designator,
      ConstantSubscript which) {
    return common::visit(
        common::visitors{
            [&](const Substring &ss) {
              if (const auto *dataRef{ss.GetParentIf<DataRef>()}) {
                if (auto result{FoldDesignator(*dataRef, which)}) {
                  if (auto start{ToInt64(ss.lower())}) {
                    std::optional<ConstantSubscript> end;
                    auto len{dataRef->LEN()};
                    if (ss.upper()) {
                      end = ToInt64(*ss.upper());
                    } else if (len) {
                      end = ToInt64(*len);
                    }
                    if (end) {
                      if (*start < 1) {
                        isOutOfRange_ = true;
                      }
                      result->Augment(KIND * (*start - 1));
                      result->set_size(
                          *end >= *start ? KIND * (*end - *start + 1) : 0);
                      if (len) {
                        if (auto lenVal{ToInt64(*len)}) {
                          if (*end > *lenVal) {
                            isOutOfRange_ = true;
                          }
                        }
                      }
                      return result;
                    }
                  }
                }
              }
              return std::optional<OffsetSymbol>{};
            },
            [&](const auto &x) { return FoldDesignator(x, which); },
        },
        designator.u);
  }

  FoldingContext &context_;
  bool getLastComponent_{false};
  ConstantSubscript elementNumber_{0}; // zero-based
  bool isEmpty_{false};
  bool isOutOfRange_{false};
};

// Reconstructs a Designator<> from a symbol and an offset.
std::optional<Expr<SomeType>> OffsetToDesignator(
    FoldingContext &, const Symbol &, ConstantSubscript offset, std::size_t);
std::optional<Expr<SomeType>> OffsetToDesignator(
    FoldingContext &, const OffsetSymbol &);

// Represents a compile-time constant Descriptor suitable for use
// as a pointer initializer.  Lower bounds are always 1.
struct ConstantObjectPointer : public OffsetSymbol {
  struct Dimension {
    ConstantSubscript byteStride;
    ConstantSubscript extent;
  };
  using Dimensions = std::vector<Dimension>;

  ConstantObjectPointer(
      const Symbol &symbol, std::size_t size, Dimensions &&dims)
      : OffsetSymbol{symbol, size}, dimensions{std::move(dims)} {}

  // Folds a designator to a constant pointer.  Crashes on failure.
  // Use IsInitialDataTarget() to validate the expression beforehand.
  static ConstantObjectPointer From(FoldingContext &, const Expr<SomeType> &);

  Dimensions dimensions;
};

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_DESIGNATOR_H_
