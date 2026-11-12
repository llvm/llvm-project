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
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <vector>

namespace Fortran::evaluate {

// The relation between the parts of an object selected by two designators.
// Disjoint is also the answer when two selections cannot be compared, for
// example because a subscript is not constant or a triplet has a stride other
// than one. It is therefore not a proof that the selections do not overlap.
enum class DesignatorRelation {
  Equal,
  Contains,
  ContainedBy,
  Overlaps,
  Disjoint,
};

class DesignatorPath {
public:
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

  // Describes how the parts selected by this path relate to those selected by
  // the other path.
  DesignatorRelation Compare(const DesignatorPath &) const;
  // Returns false only when this path certainly cannot cover the other path.
  // It is optimistic where the two cannot be compared, and is used to avoid
  // false DEFAULT(NONE) errors. A path that Compare reports as Equal to or as
  // containing the other path always may contain it.
  bool MayContain(const DesignatorPath &) const;
  std::string AsFortran() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  void SetBase(NamedEntity);
  void AddComponent(const Symbol &);
  void AddSubscripts(std::vector<Subscript>);
  const std::optional<NamedEntity> &base() const { return base_; }
  // The enclosing COMMON block, or nullptr when the base is not in COMMON.
  const Symbol *commonBlock() const { return commonBlock_; }
  const std::vector<Part> &parts() const { return parts_; }
  bool empty() const { return !base_ && parts_.empty(); }
  bool HasBaseOnly() const { return base_ && parts_.empty(); }
  bool operator==(const DesignatorPath &that) const {
    return base_ == that.base_ && commonBlock_ == that.commonBlock_ &&
        parts_ == that.parts_;
  }

private:
  bool IsWholeCommonBlock() const;

  std::optional<NamedEntity> base_;
  const Symbol *commonBlock_{nullptr};
  std::vector<Part> parts_;
};

} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_DESIGNATOR_PATH_H_
