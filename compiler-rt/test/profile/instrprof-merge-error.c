// RUN: rm -rf %t; mkdir %t

// RUN: %clang_pgogen -o %t/timeprof -mllvm -pgo-temporal-instrumentation %s
// RUN: env LLVM_PROFILE_FILE=%t/timeprof_%m.profdata %run %t/timeprof 2>&1 | count 0
// RUN: env LLVM_PROFILE_FILE=%t/timeprof_%m.profdata %run %t/timeprof 2>&1 | FileCheck %s --check-prefix=TIMEPROF

// TIMEPROF: Temporal profiles do not support profile merging at runtime.

int main() { return 0; }
