// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

// Comments and code excluded by the preprocessor in the body of an unused
// function are marked as skipped, just like in used functions. Otherwise, they
// would be reported as uncovered lines if the unused function's mapping is
// used on its own, e.g. by llvm-cov with -empty-profile.

// CHECK-LABEL: {{_Z6unusedi|\?unused@@YAHH@Z}}:
// CHECK-NEXT: File 0, [[@LINE+4]]:26 -> [[@LINE+11]]:2 = 0
// CHECK-NEXT: Skipped,File 0, [[@LINE+4]]:1 -> [[@LINE+4]]:16 = 0
// CHECK-NEXT: Skipped,File 0, [[@LINE+5]]:1 -> [[@LINE+7]]:7 = 0
// CHECK-NOT: Skipped
inline int unused(int x) {
  // A comment.
  ++x;
#if 0
  ++x;
#endif
  return x;
}
