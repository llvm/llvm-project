// XFAIL: target={{.*}}-aix{{.*}}
// Test that the instrumentation puts the right linkage on the profile data for
// inline functions.
// RUN: %clang_profgen -g -fcoverage-mapping -c -o %t1.o %s -DOBJECT_1
// RUN: %clang_profgen -g -fcoverage-mapping -c -o %t2.o %s
// RUN: %clang_profgen -g -fcoverage-mapping %t1.o %t2.o -o %t.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.exe
// RUN: llvm-profdata show %t.profraw -all-functions | FileCheck %s

// Again, with optimizations and inlining. This tests that we use comdats
// correctly.
// RUN: %clang_profgen -O2 -g -fcoverage-mapping -c -o %t1.o %s -DOBJECT_1
// RUN: %clang_profgen -O2 -g -fcoverage-mapping -c -o %t2.o %s
// RUN: %clang_profgen -g -fcoverage-mapping %t1.o %t2.o -o %t.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.exe
// RUN: llvm-profdata show %t.profraw -all-functions | FileCheck %s

// CHECK:  {{.*}}foo{{.*}}:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1
// CHECK:  {{.*}}inline_wrapper{{.*}}:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 2
// CHECK:  main:
// CHECK-NEXT:    Hash:
// CHECK-NEXT:    Counters: 1
// CHECK-NEXT:    Function count: 1

extern "C" int puts(const char *);

inline void inline_wrapper(const char *msg) {
  puts(msg);
}

void foo();

#ifdef OBJECT_1
void foo() {
  inline_wrapper("foo");
}
#else
int main() {
  inline_wrapper("main");
  foo();
}
#endif
