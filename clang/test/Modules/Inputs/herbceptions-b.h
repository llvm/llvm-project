#pragma once
#include "herbceptions-a.h"

// Cross-module reference: defined in herbc_b, calls a declaration from herbc_a.
inline int wrap_fail(int x) return_failure{int} {
  auto e = catch return_failure(fail_fn(x));
  return e.failed ? e.error : e.value;
}
