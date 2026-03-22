//===-- lib/Semantics/openmp-utils.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities used in OpenMP semantic checks.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_UTILS_H
#define FORTRAN_SEMANTICS_OPENMP_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/message.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"

#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Fortran::semantics {
class Scope;
class SemanticsContext;
class Symbol;

// Add this namespace to avoid potential conflicts
namespace omp {
using Fortran::parser::omp::BlockRange;
using Fortran::parser::omp::ExecutionPartIterator;
using Fortran::parser::omp::is_range_v;
using Fortran::parser::omp::LoopNestIterator;
using Fortran::parser::omp::LoopRange;

template <typename T, typename U = std::remove_const_t<T>> U AsRvalue(T &t) {
  return U(t);
}

template <typename T> T &&AsRvalue(T &&t) { return std::move(t); }

const Scope &GetScopingUnit(const Scope &scope);
const Scope &GetProgramUnit(const Scope &scope);

// There is no consistent way to get the source of an ActionStmt, but there
// is "source" in Statement<T>. This structure keeps the ActionStmt with the
// extracted source for further use.
struct SourcedActionStmt {
  const parser::ActionStmt *stmt{nullptr};
  parser::CharBlock source;

  operator bool() const { return stmt != nullptr; }
};

SourcedActionStmt GetActionStmt(const parser::ExecutionPartConstruct *x);
SourcedActionStmt GetActionStmt(const parser::Block &block);

std::string ThisVersion(unsigned version);
std::string TryVersion(unsigned version);

const parser::Designator *GetDesignatorFromObj(const parser::OmpObject &object);
const parser::DataRef *GetDataRefFromObj(const parser::OmpObject &object);
const parser::ArrayElement *GetArrayElementFromObj(
    const parser::OmpObject &object);
const Symbol *GetObjectSymbol(const parser::OmpObject &object);
std::optional<parser::CharBlock> GetObjectSource(
    const parser::OmpObject &object);
const Symbol *GetArgumentSymbol(const parser::OmpArgument &argument);
const parser::OmpObject *GetArgumentObject(const parser::OmpArgument &argument);

bool IsCommonBlock(const Symbol &sym);
bool IsExtendedListItem(const Symbol &sym);
bool IsVariableListItem(const Symbol &sym);
bool IsTypeParamInquiry(const Symbol &sym);
bool IsStructureComponent(const Symbol &sym);
bool IsVarOrFunctionRef(const MaybeExpr &expr);

bool IsWholeAssumedSizeArray(const parser::OmpObject &object);

bool IsMapEnteringType(parser::OmpMapType::Value type);
bool IsMapExitingType(parser::OmpMapType::Value type);

MaybeExpr GetEvaluateExpr(const parser::Expr &parserExpr);
template <typename T> MaybeExpr GetEvaluateExpr(const T &inp) {
  return GetEvaluateExpr(parser::UnwrapRef<parser::Expr>(inp));
}

std::optional<evaluate::DynamicType> GetDynamicType(
    const parser::Expr &parserExpr);

std::optional<bool> GetLogicalValue(const SomeExpr &expr);

std::optional<bool> IsContiguous(
    SemanticsContext &semaCtx, const parser::OmpObject &object);

std::vector<SomeExpr> GetAllDesignators(const SomeExpr &expr);
const SomeExpr *HasStorageOverlap(
    const SomeExpr &base, llvm::ArrayRef<SomeExpr> exprs);
bool IsAssignment(const parser::ActionStmt *x);
bool IsPointerAssignment(const evaluate::Assignment &x);

MaybeExpr MakeEvaluateExpr(const parser::OmpStylizedInstance &inp);

/// A representation of a "because" message.
struct Reason {
  Reason() = default;
  Reason(Reason &&) = default;
  Reason(const Reason &);
  Reason &operator=(Reason &&) = default;
  Reason &operator=(const Reason &);

  parser::Messages msgs;

  template <typename... Ts> Reason &Say(Ts &&...args) {
    msgs.Say(std::forward<Ts>(args)...);
    return *this;
  }
  parser::Message &AttachTo(parser::Message &msg);
  Reason &Append(const Reason &other) {
    CopyFrom(other);
    return *this;
  }
  operator bool() const { return !msgs.empty(); }

private:
  void CopyFrom(const Reason &other);
};

// A property with an explanation of its value. Both, the property and the
// reason are optional (the reason can have no messages in it).
template <typename T> struct WithReason {
  std::optional<T> value;
  Reason reason;

  WithReason() = default;
  WithReason(std::optional<T> v, const Reason &r = Reason())
      : value(v), reason(r) {}
  operator bool() const { return value.has_value(); }
};

WithReason<int64_t> GetArgumentValueWithReason(
    const parser::OmpDirectiveSpecification &spec, llvm::omp::Clause clauseId,
    unsigned version);
WithReason<int64_t> GetNumArgumentsWithReason(
    const parser::OmpDirectiveSpecification &spec, llvm::omp::Clause clauseId,
    unsigned version);

bool IsLoopTransforming(llvm::omp::Directive dir);
bool IsFullUnroll(const parser::OpenMPLoopConstruct &x);

// Return the depth of the affected nests:
//   {affected-depth, reason, must-be-perfect-nest}.
std::pair<WithReason<int64_t>, bool> GetAffectedNestDepthWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version);
// Return the range of the affected nests in the sequence:
//   {first, count, reason}.
// If the range is "the whole sequence", the return value will be {1, -1, ...}.
WithReason<std::pair<int64_t, int64_t>> GetAffectedLoopRangeWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version);

// Count the required loop count from range. If count == -1, return -1,
// indicating all loops in the sequence.
std::optional<int64_t> GetRequiredCount(
    std::optional<int64_t> first, std::optional<int64_t> count);
std::optional<int64_t> GetRequiredCount(
    std::optional<std::pair<int64_t, int64_t>> range);

struct LoopSequence {
  LoopSequence(const parser::ExecutionPartConstruct &root, unsigned version,
      bool allowAllLoops = false);

  template <typename R, typename = std::enable_if_t<is_range_v<R>>>
  LoopSequence(const R &range, unsigned version, bool allowAllLoops = false)
      : version_(version), allowAllLoops_(allowAllLoops) {
    entry_ = std::make_unique<Construct>(range, nullptr);
    createChildrenFromRange(entry_->location);
    precalculate();
  }

  struct Depth {
    // If this sequence is a nest, the depth of the Canonical Loop Nest rooted
    // at this sequence. Otherwise unspecified.
    std::optional<int64_t> semantic;
    // If this sequence is a nest, the depth of the perfect Canonical Loop Nest
    // rooted at this sequence. Otherwise unspecified.
    std::optional<int64_t> perfect;
  };

  bool isNest() const { return length_ && *length_ == 1; }
  std::optional<int64_t> length() const { return length_; }
  const Depth &depth() const { return depth_; }
  const std::vector<LoopSequence> &children() const { return children_; }

private:
  using Construct = ExecutionPartIterator::Construct;

  LoopSequence(
      std::unique_ptr<Construct> entry, unsigned version, bool allowAllLoops);

  template <typename R, typename = std::enable_if_t<is_range_v<R>>>
  void createChildrenFromRange(const R &range) {
    createChildrenFromRange(range.begin(), range.end());
  }

  std::unique_ptr<Construct> createConstructEntry(
      const parser::ExecutionPartConstruct &code);

  void createChildrenFromRange( //
      ExecutionPartIterator::IteratorType begin,
      ExecutionPartIterator::IteratorType end);

  /// Precalculate length and depth.
  void precalculate();

  std::optional<int64_t> calculateLength() const;
  std::optional<int64_t> getNestedLength() const;
  Depth calculateDepths() const;
  Depth getNestedDepths() const;

  /// The construct that is not a loop or a loop-transforming construct,
  /// that is also not a valid intervening code. Unset if no such code is
  /// present.
  const parser::ExecutionPartConstruct *invalidIC_{nullptr};
  /// The construct that is not a loop or a loop-transforming construct,
  /// whose presence would prevent perfect nesting of loops (i.e. code that
  /// is not "transparent" to a perfect nest).
  const parser::ExecutionPartConstruct *opaqueIC_{nullptr};

  /// Precalculated length of the sequence. Note that this is different from
  /// the number of children because a child may result in a sequence, for
  /// example a fuse with a reduced loop range. The length of that sequence
  /// adds to the length of the owning LoopSequence.
  std::optional<int64_t> length_;
  /// Precalculated depths. Only meaningful if the sequence is a nest.
  Depth depth_;

  // The core structure of the class:
  unsigned version_; // Needed for GetXyzWithReason
  bool allowAllLoops_;
  std::unique_ptr<Construct> entry_;
  std::vector<LoopSequence> children_;
};
} // namespace omp
} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_OPENMP_UTILS_H
