#include "src/math/log.h"
#include "src/__support/math/log.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(double, log, (double x)) { return math::log(x); }

} // namespace LIBC_NAMESPACE_DECL
