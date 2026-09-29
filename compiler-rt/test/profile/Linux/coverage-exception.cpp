// REQUIRES: lld-available
// XFAIL: powerpc64-target-arch

// RUN: %clangxx_profgen -std=c++17 -fuse-ld=lld -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile=%t.profdata 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#define TRY_AND_CATCH_ALL(x)                                                   \
  try {                                                                        \
    (x);                                                                       \
  } catch (...) {                                                              \
  }

#define TRY_MAYBE_CRASH(x)                                                     \
  try {                                                                        \
    if ((x)) {                                                                 \
      printf("no crash\n");                                                    \
    } else {                                                                   \
      abort();                                                                 \
    }                                                                          \
  } catch (...) {                                                              \
  }

#define TRY_AND_CATCH_CRASHES(x)                                               \
  try {                                                                        \
    (x);                                                                       \
  } catch (...) {                                                              \
    abort();                                                                   \
  }

static __attribute__((noinline)) int do_throw(bool b) {
  if (b)
    throw b;
  return 1;
}

// clang-format off
static
int test_no_exception() {               // CHECK:  [[@LINE]]| 1|int test_no_exception()
  try {                                 // CHECK:  [[@LINE]]| 1|  try {
    do_throw(false);                    // CHECK:  [[@LINE]]| 1|    do_throw(
  } catch (...) {                       // CHECK:  [[@LINE]]| 1|  } catch (
    abort();                            // CHECK:  [[@LINE]]| 0|    abort(
  }                                     // CHECK:  [[@LINE]]| 0|  }
  printf("%s\n", __func__);             // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_no_exception_macro() {         // CHECK:  [[@LINE]]| 1|int test_no_exception_macro()
  TRY_AND_CATCH_ALL(do_throw(false));   // CHECK:  [[@LINE]]| 1|  TRY_AND_CATCH_ALL(
  printf("%s\n", __func__);             // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_exception() {                  // CHECK:  [[@LINE]]| 1|int test_exception()
  try {                                 // CHECK:  [[@LINE]]| 1|  try {
    do_throw(true);                     // CHECK:  [[@LINE]]| 1|    do_throw(
  } catch (...) {                       // CHECK:  [[@LINE]]| 1|  } catch (
    printf("%s\n", __func__);           // CHECK:  [[@LINE]]| 1|    printf(
  }                                     // CHECK:  [[@LINE]]| 1|  }
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_exception_macro() {            // CHECK:  [[@LINE]]| 1|int test_exception_macro()
  TRY_AND_CATCH_ALL(do_throw(true));    // CHECK:  [[@LINE]]| 1|  TRY_AND_CATCH_ALL(
  printf("%s\n", __func__);             // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_exception_macro_nested() {     // CHECK:  [[@LINE]]| 1|int test_exception_macro_nested()
  try {                                 // CHECK:  [[@LINE]]| 1|  try {
    TRY_AND_CATCH_ALL(do_throw(true));  // CHECK:  [[@LINE]]| 1|    TRY_AND_CATCH_ALL(
  } catch (...) {                       // CHECK:  [[@LINE]]| 1|  } catch (
    abort();                            // CHECK:  [[@LINE]]| 0|    abort(
  }                                     // CHECK:  [[@LINE]]| 0|  }
  printf("%s\n", __func__);             // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_exception_try_crash() {        // CHECK:  [[@LINE]]| 1|int test_exception_try_crash()
  TRY_MAYBE_CRASH(do_throw(false));     // CHECK:  [[@LINE]]| 1|  TRY_MAYBE_CRASH(
  printf("%s\n", __func__);             // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_exception_crash() {            // CHECK:  [[@LINE]]| 1|int test_exception_crash()
  TRY_AND_CATCH_CRASHES(do_throw(false)); // CHECK:  [[@LINE]]| 1|  TRY_AND_CATCH_CRASHES(
  printf("%s\n", __func__);             // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

static
int test_conditional(int i) {           // CHECK:  [[@LINE]]| 1|int test_conditional(int i)
  try {                                 // CHECK:  [[@LINE]]| 1|  try {
    if (i % 2 == 0) {                   // CHECK:  [[@LINE]]| 1|    if (
      printf("%s\n", __func__);         // CHECK:  [[@LINE]]| 1|      printf(
    } else {                            // CHECK:  [[@LINE]]| 1|    } else {
      do_throw(true);                   // CHECK:  [[@LINE]]| 0|      do_throw(
    }                                   // CHECK:  [[@LINE]]| 0|    }
  } catch (...) {                       // CHECK:  [[@LINE]]| 1|  } catch (
    abort();                            // CHECK:  [[@LINE]]| 0|    abort(
  }                                     // CHECK:  [[@LINE]]| 0|  }
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}

static
int test_multiple_catch() {             // CHECK:  [[@LINE]]| 1|int test_multiple_catch()
  try {                                 // CHECK:  [[@LINE]]| 1|  try {
    do_throw(true);                     // CHECK:  [[@LINE]]| 1|    do_throw(
  } catch (double) {                    // CHECK:  [[@LINE]]| 1|  } catch (double)
    abort();                            // CHECK:  [[@LINE]]| 0|    abort(
  } catch (bool) {                      // CHECK:  [[@LINE]]| 1|  } catch (bool)
    printf("bool\n");                   // CHECK:  [[@LINE]]| 1|    printf(
  } catch (float) {                     // CHECK:  [[@LINE]]| 1|  } catch (float)
    abort();                            // CHECK:  [[@LINE]]| 0|    abort(
  } catch (...) {                       // CHECK:  [[@LINE]]| 0|  } catch (
    abort();                            // CHECK:  [[@LINE]]| 0|    abort(
  }                                     // CHECK:  [[@LINE]]| 0|  }
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}

int main() {                            // CHECK:  [[@LINE]]| 1|int main()
  test_no_exception();                  // CHECK:  [[@LINE]]| 1|  test_no_exception(
  test_no_exception_macro();            // CHECK:  [[@LINE]]| 1|  test_no_exception_macro(
  test_exception();                     // CHECK:  [[@LINE]]| 1|  test_exception(
  test_exception_macro();               // CHECK:  [[@LINE]]| 1|  test_exception_macro(
  test_exception_macro_nested();        // CHECK:  [[@LINE]]| 1|  test_exception_macro_nested(
  test_exception_try_crash();           // CHECK:  [[@LINE]]| 1|  test_exception_try_crash(
  test_exception_crash();               // CHECK:  [[@LINE]]| 1|  test_exception_crash(
  test_conditional(2);                  // CHECK:  [[@LINE]]| 1|  test_conditional(
  test_multiple_catch();                // CHECK:  [[@LINE]]| 1|  test_multiple_catch(
  return 0;                             // CHECK:  [[@LINE]]| 1|  return
}                                       // CHECK:  [[@LINE]]| 1|}
// clang-format on
