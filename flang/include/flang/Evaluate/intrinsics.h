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
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
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

class IntrinsicProcTable {
private:
  class Implementation;

public:
  ~IntrinsicProcTable();
  static IntrinsicProcTable Configure(
      const common::IntrinsicTypeDefaultKinds &);

  // Check whether a name should be allowed to appear on an INTRINSIC
  // statement.
  bool IsIntrinsic(const std::string &) const;

  // Probe the intrinsics for a match against a specific call.
  // On success, the actual arguments are transferred to the result
  // in dummy argument order; on failure, the actual arguments remain
  // untouched.
  std::optional<SpecificCall> Probe(
      const CallCharacteristics &, ActualArguments &, FoldingContext &) const;

  // Probe the intrinsics with the name of a potential specific intrinsic.
  std::optional<SpecificIntrinsicFunctionInterface> IsSpecificIntrinsicFunction(
      const std::string &) const;

  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  Implementation *impl_{nullptr};  // owning pointer
};
}
#endif  // FORTRAN_EVALUATE_INTRINSICS_H_
