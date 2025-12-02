// UNSUPPORTED: system-windows

// Tests to check that we pass -L <install>/bin/../lib/ to the linker to prioritize the toolchain's
// libc++.dylib over the system's libc++.dylib on Darwin. This matches the behavior we have for
// header search paths, where we prioritize toolchain headers and then fall back to the sysroot ones.

// Check that we pass the right -L to the linker even when -stdlib=libc++ is not passed.
//
// RUN: %clang -### %s 2>&1                                                     \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx    \
// RUN:               --check-prefix=CHECK-1 %s
//
// CHECK-1: "/usr/bin/ld"
// CHECK-1: "-L" "[[TOOLCHAIN]]/usr/bin/../lib"

// Check that we pass the right -L to the linker when -stdlib=libc++ is passed, both in the
// case where there is libc++.dylib in the toolchain and when there isn't.
//
// RUN: %clang -### %s 2>&1                                                     \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     -stdlib=libc++                                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx    \
// RUN:               --check-prefix=CHECK-2 %s
//
// CHECK-2: "/usr/bin/ld"
// CHECK-2: "-L" "[[TOOLCHAIN]]/usr/bin/../lib"
//
// RUN: %clang -### %s 2>&1                                             \
// RUN:     --target=x86_64-apple-darwin                                \
// RUN:     -stdlib=libc++                                              \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin   \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain      \
// RUN:               --check-prefix=CHECK-3 %s
//
// CHECK-3: "/usr/bin/ld"
// CHECK-3: "-L" "[[TOOLCHAIN]]/usr/bin/../lib"
