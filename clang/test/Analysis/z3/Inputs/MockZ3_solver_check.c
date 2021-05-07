#include <dlfcn.h>
#include <stdio.h>

#include <z3.h>

// Mock implementation: return UNDEF for the 5th invocation, otherwise it just
// returns the result of the real invocation.
Z3_lbool Z3_API Z3_solver_check(Z3_context c, Z3_solver s) {
  static Z3_lbool (*OriginalFN)(Z3_context, Z3_solver);
  if (!OriginalFN) {
    OriginalFN = (Z3_lbool(*)(Z3_context, Z3_solver))dlsym(RTLD_NEXT,
                                                           "Z3_solver_check");
  }

  // Invoke the actual solver.
  Z3_lbool Result = OriginalFN(c, s);

  // Mock the 5th invocation to return UNDEF.
  static unsigned int Counter = 0;
  if (++Counter == 5) {
    fprintf(stderr, "Z3_solver_check returns a mocked value: UNDEF\n");
    return Z3_L_UNDEF;
  }
  fprintf(stderr, "Z3_solver_check returns the real value: %s\n",
          (Result == Z3_L_UNDEF ? "UNDEF"
                                : ((Result == Z3_L_TRUE ? "TRUE" : "FALSE"))));
  return Result;
}
