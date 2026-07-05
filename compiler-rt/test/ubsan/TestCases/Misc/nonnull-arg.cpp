// RUN: %clangxx -fsanitize=nonnull-attribute -fno-sanitize-recover=all %s -O3 -o %t
// RUN: %run %t nc
// RUN: %run %t nm
// RUN: %run %t nf
// RUN: %run %t nv
// RUN: not %run %t 0c 2>&1 | FileCheck %s --check-prefix=CTOR
// RUN: not %run %t 0m 2>&1 | FileCheck %s --check-prefix=METHOD
// RUN: not %run %t 0f 2>&1 | FileCheck %s --check-prefix=FUNC
// RUN: not %run %t 0v 2>&1 | FileCheck %s --check-prefix=VARIADIC
//
// AArch64 lacks variadic instrumentation for MSAN.
// REQUIRES: stable-runtime

class C {
  int *null_;
  int *nonnull_;

public:
  C(int *null, __attribute__((nonnull)) int *nonnull)
      : null_(null), nonnull_(nonnull) {}
  int value() { return *nonnull_; }
  int method(int *nonnull, int *null) __attribute__((nonnull(2))) {
    return *nonnull_ + *nonnull;
  }
};

__attribute__((nonnull)) int func(int *nonnull) { return *nonnull; }

#include <stdarg.h>
__attribute__((nonnull)) int variadic(int x, ...) {
  va_list args;
  va_start(args, x);
  int *nonnull = va_arg(args, int*);
  int res = *nonnull;
  va_end(args);
  return res;
}

int main(int argc, char *argv[]) {
  int local = 0;
  int *arg = (argv[1][0] == '0') ? 0x0 : &local;
  switch (argv[1][1]) {
    case 'c':
      return C(0x0, arg).value();
      // CTOR: {{.*}}nonnull-arg.cpp:[[@LINE-1]]:21: runtime error: null pointer passed as argument 2, which is declared to never be null
      // CTOR-NEXT: {{.*}}nonnull-arg.cpp:19:31: note: nonnull attribute specified here
    case 'm':
      return C(0x0, &local).method(arg, 0x0);
      // METHOD: {{.*}}nonnull-arg.cpp:[[@LINE-1]]:36: runtime error: null pointer passed as argument 1, which is declared to never be null
      // METHOD-NEXT: {{.*}}nonnull-arg.cpp:22:54: note: nonnull attribute specified here
    case 'f':
      return func(arg);
      // FUNC: {{.*}}nonnull-arg.cpp:[[@LINE-1]]:19: runtime error: null pointer passed as argument 1, which is declared to never be null
      // FUNC-NEXT: {{.*}}nonnull-arg.cpp:27:16: note: nonnull attribute specified here
    case 'v':
      return variadic(42, arg);
    // VARIADIC: {{.*}}nonnull-arg.cpp:[[@LINE-1]]:27: runtime error: null pointer passed as argument 2, which is declared to never be null
    // VARIADIC-NEXT: {{.*}}nonnull-arg.cpp:30:16: note: nonnull attribute specified here
  }
  return 0;
}
