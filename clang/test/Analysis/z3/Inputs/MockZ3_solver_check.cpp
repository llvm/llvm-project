#include <cassert>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <z3.h>

static char *Z3ResultsBegin;
static char *Z3ResultsCursor;

static __attribute__((constructor)) void init() {
  const char *Env = getenv("Z3_SOLVER_RESULTS");
  if (!Env) {
    fprintf(stderr, "Z3_SOLVER_RESULTS envvar must be defined; abort\n");
    abort();
  }
  Z3ResultsBegin = strdup(Env);
  Z3ResultsCursor = Z3ResultsBegin;
  if (!Z3ResultsBegin) {
    fprintf(stderr, "strdup failed; abort\n");
    abort();
  }
}

static __attribute__((destructor)) void finit() {
  if (strlen(Z3ResultsCursor) > 0) {
    fprintf(stderr, "Z3_SOLVER_RESULTS should have been completely consumed "
                    "by the end of the test; abort\n");
    abort();
  }
  free(Z3ResultsBegin);
}

static bool consume_token(char **pointer_to_cursor, const char *token) {
  assert(pointer_to_cursor);
  int len = strlen(token);
  if (*pointer_to_cursor && strncmp(*pointer_to_cursor, token, len) == 0) {
    *pointer_to_cursor += len;
    return true;
  }
  return false;
}

Z3_lbool Z3_API Z3_solver_check(Z3_context c, Z3_solver s) {
  consume_token(&Z3ResultsCursor, ",");

  if (consume_token(&Z3ResultsCursor, "UNDEF")) {
    printf("Z3_solver_check returns UNDEF\n");
    return Z3_L_UNDEF;
  }
  if (consume_token(&Z3ResultsCursor, "SAT")) {
    printf("Z3_solver_check returns SAT\n");
    return Z3_L_TRUE;
  }
  if (consume_token(&Z3ResultsCursor, "UNSAT")) {
    printf("Z3_solver_check returns UNSAT\n");
    return Z3_L_FALSE;
  }
  fprintf(stderr, "Z3_SOLVER_RESULTS was exhausted; abort\n");
  abort();
}
