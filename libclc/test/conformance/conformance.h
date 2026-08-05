//===----------------------------------------------------------------------===//
//
// Shared helpers for libclc execution conformance tests. These kernels are
// launched on-device and report failure by trapping.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCLC_TEST_CONFORMANCE_H
#define LIBCLC_TEST_CONFORMANCE_H

// Yields an input value the compiler cannot optimize out.
#define TEST_INPUT(TYPE, VALUE)                                                \
  ({                                                                           \
    volatile TYPE __clc_in = (VALUE);                                          \
    __clc_in;                                                                  \
  })

// Check the condition and submit a hardware trap on failure.
#define CHECK(COND)                                                            \
  do {                                                                         \
    if (!(COND))                                                               \
      __builtin_verbose_trap("libclc", "check failed: " #COND);                \
  } while (0)

#define CHECK_EQ(LHS, RHS) CHECK((LHS) == (RHS))
#define CHECK_NE(LHS, RHS) CHECK((LHS) != (RHS))
#define CHECK_LT(LHS, RHS) CHECK((LHS) < (RHS))
#define CHECK_LE(LHS, RHS) CHECK((LHS) <= (RHS))
#define CHECK_GT(LHS, RHS) CHECK((LHS) > (RHS))
#define CHECK_GE(LHS, RHS) CHECK((LHS) >= (RHS))

#endif // LIBCLC_TEST_CONFORMANCE_H
