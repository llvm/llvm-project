// UNSUPPORTED: system-windows

// Tests to check that we link against the toolchain-provided libc++ built library when it is provided.
// This is required to prefer the toolchain's libc++ over the system's libc++, which matches the behavior
// we have for header search paths.
//
// Note that we explicitly specify the linker path to use to make this test portable across platforms.

// When libc++.dylib is NOT in the toolchain, we should use -lc++ and fall back to the libc++
// in the sysroot.
//
// (1) Without -fexperimental-library.
// RUN: %clangxx -### %s 2>&1                                                   \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     --ld-path=%S/Inputs/lld/ld.lld                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx    \
// RUN:               -DINPUTS=%S/Inputs                                        \
// RUN:               --check-prefix=CHECK-1 %s
// CHECK-1: "[[INPUTS]]/lld/ld.lld"
// CHECK-1: "-lc++"
// CHECK-1-NOT: "[[TOOLCHAIN]]/usr/lib"
//
// (2) With -fexperimental-library.
// RUN: %clangxx -### %s 2>&1                                                   \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     --ld-path=%S/Inputs/lld/ld.lld                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_no_libcxx/usr/bin \
// RUN:     -fexperimental-library                                              \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_no_libcxx    \
// RUN:               -DINPUTS=%S/Inputs                                        \
// RUN:               --check-prefix=CHECK-2 %s
// CHECK-2: "[[INPUTS]]/lld/ld.lld"
// CHECK-2: "-lc++" "-lc++experimental"
// CHECK-2-NOT: "[[TOOLCHAIN]]/usr/lib"

// When we have libc++.dylib in the toolchain, it should be used over the one in the sysroot.
// There are a few cases worth testing.
//
// (1) Without -fexperimental-library.
// RUN: %clangxx -### %s 2>&1                                                   \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     --ld-path=%S/Inputs/lld/ld.lld                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin           \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain              \
// RUN:               -DINPUTS=%S/Inputs                                        \
// RUN:               --check-prefix=CHECK-3 %s
// CHECK-3: "[[INPUTS]]/lld/ld.lld"
// CHECK-3: "[[TOOLCHAIN]]/usr/lib/libc++.dylib"
// CHECK-3-NOT: "-lc++"
//
// (2) With -fexperimental-library.
// RUN: %clangxx -### %s 2>&1                                                   \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     --ld-path=%S/Inputs/lld/ld.lld                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain/usr/bin           \
// RUN:     -fexperimental-library                                              \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain              \
// RUN:               -DINPUTS=%S/Inputs                                        \
// RUN:               --check-prefix=CHECK-4 %s
// CHECK-4: "[[INPUTS]]/lld/ld.lld"
// CHECK-4: "[[TOOLCHAIN]]/usr/lib/libc++.dylib"
// CHECK-4: "[[TOOLCHAIN]]/usr/lib/libc++experimental.a"
// CHECK-4-NOT: "-lc++"
// CHECK-4-NOT: "-lc++experimental"

// When we have libc++.a in the toolchain instead of libc++.dylib, it should be
// used over the one in the sysroot.
//
// (1) Without -fexperimental-library.
// RUN: %clangxx -### %s 2>&1                                                   \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     --ld-path=%S/Inputs/lld/ld.lld                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_static/usr/bin    \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_static       \
// RUN:               -DINPUTS=%S/Inputs                                        \
// RUN:               --check-prefix=CHECK-5 %s
// CHECK-5: "[[INPUTS]]/lld/ld.lld"
// CHECK-5: "[[TOOLCHAIN]]/usr/lib/libc++.a"
// CHECK-5-NOT: "-lc++"
//
// (2) With -fexperimental-library.
// RUN: %clangxx -### %s 2>&1                                                   \
// RUN:     --target=x86_64-apple-darwin                                        \
// RUN:     --ld-path=%S/Inputs/lld/ld.lld                                      \
// RUN:     -ccc-install-dir %S/Inputs/basic_darwin_toolchain_static/usr/bin    \
// RUN:     -fexperimental-library                                              \
// RUN:   | FileCheck -DTOOLCHAIN=%S/Inputs/basic_darwin_toolchain_static       \
// RUN:               -DINPUTS=%S/Inputs                                        \
// RUN:               --check-prefix=CHECK-6 %s
// CHECK-6: "[[INPUTS]]/lld/ld.lld"
// CHECK-6: "[[TOOLCHAIN]]/usr/lib/libc++.a"
// CHECK-6: "[[TOOLCHAIN]]/usr/lib/libc++experimental.a"
// CHECK-6-NOT: "-lc++"
// CHECK-6-NOT: "-lc++experimental"
