#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP

#include <cstdint>
#include <string>

namespace transpiler {

// Structured reason for a raise failure. Lives in its own header so the
// handler layer (`raise_context.hpp`) can depend on failure values
// without pulling in `RaiseResult` and the rest of the top-level
// `raiser.hpp` interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
};

const char *reasonString(RaiseFailureReason r);

struct RaiseFailure {
  RaiseFailureReason reason = RaiseFailureReason::None;
  // Optional human-readable context.
  std::string detail;

  bool hasFailed() const { return reason != RaiseFailureReason::None; }
};

} // namespace transpiler

#endif
