//===- raise_failure.h - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_H
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_H

#include <cstdint>
#include <string>

namespace COMGR::hotswap {

// Lives in its own header so the handler layer can depend on failure
// values without pulling in the rest of the top-level `raiser.h`
// interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  BadInput,
};


struct RaiseFailure {
  RaiseFailureReason Reason = RaiseFailureReason::None;
  // Optional human-readable context.
  std::string Detail;

  bool hasFailed() const { return Reason != RaiseFailureReason::None; }
};

} // namespace COMGR::hotswap

#endif
