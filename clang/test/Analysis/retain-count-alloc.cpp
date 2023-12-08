// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core,unix.Malloc \
// RUN:  -verify %s

// expected-no-diagnostics: We do not model Integer Set Library's retain-count
//                          based allocation. If any of the parameters has an
//                          '__isl_' prefixed macro definition we escape every
//                          of them when we are about to 'free()' something.

#define __isl_take
#define __isl_keep

struct Object { int Ref; };
void free(void *);

Object *copyObj(__isl_keep Object *O) {
  O->Ref++;
  return O;
}

void freeObj(__isl_take Object *O) {
  if (--O->Ref > 0)
    return;

  free(O); // Here we notice that the parameter contains '__isl_', escape it.
}

void useAfterFree(__isl_take Object *A) {
  if (!A)
    return;

  Object *B = copyObj(A);
  freeObj(B);

  A->Ref = 13;
  // no-warning: 'Use of memory after it is freed' was here.
}
