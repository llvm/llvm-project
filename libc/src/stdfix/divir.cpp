#include "divir.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/divifx.h" // divifx_impl
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h" // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, divir, (int i, fract f)) {
  return fixed_point::divir(i, f);
}

} // namespace LIBC_NAMESPACE_DECL
