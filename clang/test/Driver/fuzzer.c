// Test flags inserted by -fsanitize=fuzzer.

// RUN: %clang -fsanitize=fuzzer %s -target x86_64-apple-darwin14 -### 2>&1 | FileCheck --check-prefixes=CHECK-FUZZER-LIB,CHECK-COVERAGE %s

// CHECK-COVERAGE:      -fsanitize-coverage-indirect-calls
// CHECK-COVERAGE-SAME: -fsanitize-coverage-trace-cmp
// CHECK-COVERAGE-SAME: -fsanitize-coverage-inline-8bit-counters
// CHECK-COVERAGE-SAME: -fsanitize-coverage-pc-table
// CHECK-FUZZER-LIB: libclang_rt.fuzzer

// RUN: %clang -fsanitize=fuzzer -target i386-unknown-linux -stdlib=platform %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LIBCXX-LINUX %s

// CHECK-LIBCXX-LINUX: -lstdc++

// RUN: %clang -target x86_64-apple-darwin14 -fsanitize=fuzzer %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LIBCXX-DARWIN %s

// CHECK-LIBCXX-DARWIN: -lc++


// Check that we don't link in libFuzzer.a when producing a shared object.
// RUN: %clang -fsanitize=fuzzer %s -shared -o %t.so -### 2>&1 | FileCheck --check-prefixes=CHECK-NOLIB-SO %s
// CHECK-NOLIB-SO-NOT: libclang_rt.libfuzzer

// Check that we don't link in libFuzzer when compiling with -fsanitize=fuzzer-no-link.
// RUN: %clang -fsanitize=fuzzer-no-link %s -target x86_64-apple-darwin14 -### 2>&1 | FileCheck --check-prefixes=CHECK-NOLIB,CHECK-COV %s
// CHECK-NOLIB-NOT: libclang_rt.libfuzzer
// CHECK-COV: -fsanitize-coverage-inline-8bit-counters

// RUN: %clang -fsanitize=fuzzer -fsanitize-coverage=trace-pc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-MSG %s
// CHECK-MSG-NOT: argument unused during compilation

// Check that we respect whether thes tandard library should be linked
// statically.

// RUN: %clang -fsanitize=fuzzer -target i386-unknown-linux -stdlib=libstdc++ %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LIBSTDCXX-DYNAMIC %s
// CHECK-LIBSTDCXX-DYNAMIC-NOT: -Bstatic
// CHECK-LIBSTDCXX-DYNAMIC: -lstdc++

// RUN: %clang -fsanitize=fuzzer -target i386-unknown-linux -stdlib=libstdc++ -static-libstdc++ %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LIBSTDCXX-STATIC %s
// CHECK-LIBSTDCXX-STATIC: "-Bstatic" "-lstdc++"

// RUN: %clang -fsanitize=fuzzer -target i386-unknown-linux -stdlib=libc++ %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LIBCXX-DYNAMIC %s
// CHECK-LIBCXX-DYNAMIC-NOT: -Bstatic
// CHECK-LIBCXX-DYNAMIC: -lc++

// RUN: %clang -fsanitize=fuzzer -target i386-unknown-linux -stdlib=libc++ -static-libstdc++ %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LIBCXX-STATIC %s
// CHECK-LIBCXX-STATIC: "-Bstatic" "-lc++"

int LLVMFuzzerTestOneInput(const char *Data, long Size) {
  return 0;
}
