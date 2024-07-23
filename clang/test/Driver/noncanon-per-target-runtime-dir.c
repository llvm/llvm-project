/// Check that clang's and compiler-rt's ideas of per-target runtime dirs match.

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=amd64-pc-solaris2.11 -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SOLARIS-AMD64 %s

// CHECK-SOLARIS-AMD64: x86_64-pc-solaris2.11/libclang_rt.ubsan_standalone.a
// CHECK-SOLARIS-AMD64-NOT: lib/sunos/libclang_rt.ubsan_standalone-x86_64.a"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=sparc64-unknown-linux-gnu -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/Inputs/debian_sparc64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC64 %s

// CHECK-DEBIAN-SPARC64: sparcv9-unknown-linux-gnu/libclang_rt.ubsan_standalone.a
// CHECK-DEBIAN-SPARC64-NOT: lib/linux/libclang_rt.ubsan_standalone-sparcv9.a"

