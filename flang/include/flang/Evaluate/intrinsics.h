//===-- include/flang/Evaluate/intrinsics.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTRINSICS_H_
#define FORTRAN_EVALUATE_INTRINSICS_H_

#include "call.h"
#include "characteristics.h"
#include "type.h"
#include "flang/Common/default-kinds.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/message.h"
#include <memory>
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {
class Scope;
}

namespace Fortran::evaluate {

class FoldingContext;

// Utility for checking for missing, excess, and duplicated arguments,
// and rearranging the actual arguments into dummy argument order.
bool CheckAndRearrangeArguments(ActualArguments &, parser::ContextualMessages &,
    const char *const dummyKeywords[] /* null terminated */,
    std::size_t trailingOptionals = 0);

struct CallCharacteristics {
  std::string name;
  bool isSubroutineCall{false};
};

struct SpecificCall {
  SpecificCall(SpecificIntrinsic &&si, ActualArguments &&as)
      : specificIntrinsic{std::move(si)}, arguments{std::move(as)} {}
  SpecificIntrinsic specificIntrinsic;
  ActualArguments arguments;
};

struct SpecificIntrinsicFunctionInterface : public characteristics::Procedure {
  SpecificIntrinsicFunctionInterface(
      characteristics::Procedure &&p, std::string n, bool isRestrictedSpecific)
      : characteristics::Procedure{std::move(p)}, genericName{n},
        isRestrictedSpecific{isRestrictedSpecific} {}
  std::string genericName;
  bool isRestrictedSpecific;
  // N.B. If there are multiple arguments, they all have the same type.
  // All argument and result types are intrinsic types with default kinds.
};

// Generic intrinsic classes from table 16.1
ENUM_CLASS(IntrinsicClass, atomicSubroutine, collectiveSubroutine,
    elementalFunction, elementalSubroutine, inquiryFunction, pureSubroutine,
    impureSubroutine, transformationalFunction, noClass)

class IntrinsicProcTable {
private:
  class Implementation;

  IntrinsicProcTable() = default;

public:
  ~IntrinsicProcTable();
  IntrinsicProcTable(IntrinsicProcTable &&) = default;

  static IntrinsicProcTable Configure(
      const common::IntrinsicTypeDefaultKinds &);

  // Make *this aware of the __Fortran_builtins module to expose TEAM_TYPE &c.
  void SupplyBuiltins(const semantics::Scope &) const;

  // Check whether a name should be allowed to appear on an INTRINSIC
  // statement.
  bool IsIntrinsic(const std::string &) const;
  bool IsIntrinsicFunction(const std::string &) const;
  bool IsIntrinsicSubroutine(const std::string &) const;

  // Inquiry intrinsics are defined in section 16.7, table 16.1
  IntrinsicClass GetIntrinsicClass(const std::string &) const;

  // Return the generic name of a specific intrinsic name.
  // The name provided is returned if it is a generic intrinsic name or is
  // not known to be an intrinsic.
  std::string GetGenericIntrinsicName(const std::string &) const;

  // Probe the intrinsics for a match against a specific call.
  // On success, the actual arguments are transferred to the result
  // in dummy argument order; on failure, the actual arguments remain
  // untouched.
  // For MIN and MAX, only a1 and a2 actual arguments are transferred in dummy
  // order on success and the other arguments are transferred afterwards
  // without being sorted.
  std::optional<SpecificCall> Probe(
      const CallCharacteristics &, ActualArguments &, FoldingContext &) const;

  // Probe the intrinsics with the name of a potential specific intrinsic.
  std::optional<SpecificIntrinsicFunctionInterface> IsSpecificIntrinsicFunction(
      const std::string &) const;

  // Illegal name for an intrinsic used to avoid cascading error messages when
  // constant folding.
  static const inline std::string InvalidName{
      "(invalid intrinsic function call)"};

  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  std::unique_ptr<Implementation> impl_;
};

// Check if an intrinsic explicitly allows its INTENT(OUT) arguments to be
// allocatable coarrays.
bool AcceptsIntentOutAllocatableCoarray(const std::string &);
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INTRINSICS_H_
