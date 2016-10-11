// Test that -print-libgcc-file-name correctly respects -rtlib=compiler-rt.

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name 2>&1 \
// RUN:     --target=x86_64-pc-linux \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-X8664 %s
// CHECK-CLANGRT-X8664: libclang_rt.builtins-x86_64.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name 2>&1 \
// RUN:     --target=i686-pc-linux \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-I686 %s
// CHECK-CLANGRT-I686: libclang_rt.builtins-i686.a
