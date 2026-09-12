//===-- include/flang/Evaluate/designator-path.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_DESIGNATOR_PATH_H_
#define FORTRAN_EVALUATE_DESIGNATOR_PATH_H_

#include "flang/Evaluate/expression.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Fortran::evaluate {

enum class DesignatorRelation {
  Equal,
  Contains,
  ContainedBy,
  Overlaps,
  Disjoint,
};

struct DesignatorPath {
  // A DesignatorPath represents a constrained prefix of a valid Fortran
  // designator:
  //   - an optional NamedEntity base, and
  //   - zero or more suffix parts.
  //
  // The optional base distinguishes the named entity from subsequent part
  // references. Each suffix part first applies optional subscripts to the
  // current entity and then optionally selects a component symbol. An empty
  // subscript list means there is no explicit subscript selector on this part;
  // a full slice `(:)` is represented as a single Triplet subscript with no
  // lower or upper bound and stride one. This can later grow a final optional
  // variant for terminal designator pieces that are not part refs, such as
  // complex parts, character substrings, or coarray references, while still
  // preserving a valid designator shape.
  struct Part {
    std::vector<Subscript> subscripts;
    const Symbol *symbol{nullptr};
    bool operator==(const Part &that) const {
      return subscripts == that.subscripts && symbol == that.symbol;
    }
  };

  static std::optional<DesignatorPath> Get(
      const std::optional<Expr<SomeType>> &);
  DesignatorRelation Compare(const DesignatorPath &) const;
  bool MayContain(const DesignatorPath &) const;
  std::string AsFortran() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  void SetBase(NamedEntity);
  void AddComponent(const Symbol &);
  void AddSubscripts(std::vector<Subscript>);
  const std::optional<NamedEntity> &Base() const { return base; }
  const std::vector<Part> &Parts() const { return parts; }
  bool empty() const { return !base && parts.empty(); }
  bool HasBaseOnly() const { return base && parts.empty(); }
  bool operator==(const DesignatorPath &that) const {
    return base == that.base && parts == that.parts;
  }

  struct ConstantSubscriptRange {
    std::int64_t lower;
    std::int64_t upper;
  };

  static std::optional<ConstantSubscriptRange> GetConstantSubscriptRange(
      const Subscript &);
  static bool IsFullTriplet(const Triplet &);
  static DesignatorRelation CompareSubscripts(
      const Subscript &, const Subscript &);
  static DesignatorRelation CompareSubscriptLists(
      const std::vector<Subscript> &, const std::vector<Subscript> &);
  static DesignatorRelation CompareParts(const Part &, const Part &);
  static DesignatorRelation CombineRelations(
      bool contains, bool containedBy, bool overlaps);
  static bool SubscriptMayContain(const Subscript &, const Subscript &);
  static bool SubscriptListMayContain(
      const std::vector<Subscript> &, const std::vector<Subscript> &);
  static bool PartMayContain(const Part &, const Part &);

private:
  void AddDataRef(const DataRef &);
  void AddComponent(const Component &);
  void AddNamedEntity(const NamedEntity &);
  void AddArrayRef(const ArrayRef &);
  void AddCoarrayRef(const CoarrayRef &);

  std::optional<NamedEntity> base;
  std::vector<Part> parts;
};

template <typename A> class DesignatorPathMap {
public:
  struct Entry {
    DesignatorPath path;
    A value;
  };
  using iterator = typename std::vector<Entry>::iterator;
  using const_iterator = typename std::vector<Entry>::const_iterator;

  iterator begin() { return entries_.begin(); }
  iterator end() { return entries_.end(); }
  const_iterator begin() const { return entries_.begin(); }
  const_iterator end() const { return entries_.end(); }
  bool empty() const { return entries_.empty(); }
  void clear() { entries_.clear(); }
  iterator erase(iterator iter) { return entries_.erase(iter); }
  void push_back(DesignatorPath path, A value) {
    entries_.push_back({std::move(path), std::move(value)});
  }

private:
  std::vector<Entry> entries_;
};

} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_DESIGNATOR_PATH_H_
