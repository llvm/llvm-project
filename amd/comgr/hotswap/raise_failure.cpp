#include "raise_failure.hpp"

namespace transpiler {

const char *reasonString(RaiseFailureReason r) {
  switch (r) {
  case RaiseFailureReason::None:
    return "None";
  }
  return "UnknownRaiseFailureReason";
}

} // namespace transpiler
