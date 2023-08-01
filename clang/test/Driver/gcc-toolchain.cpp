// Test that gcc-toolchain option is working correctly
//
/// Without --rtlib=libgcc the driver may pick clang_rt.crtbegin.o if
/// -DCLANG_DEFAULT_RTLIB=compiler-rt.
// RUN: %clangxx %s -### --target=x86_64-linux-gnu --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/ubuntu_14.04_multiarch_tree/usr -stdlib=libstdc++ --rtlib=libgcc --unwindlib=libgcc -no-pie 2>&1 | \
// RUN:   FileCheck %s
//
// Additionally check that the legacy spelling of the flag works.
// RUN: %clangxx %s -### --target=x86_64-linux-gnu --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/ubuntu_14.04_multiarch_tree/usr -stdlib=libstdc++ --rtlib=libgcc --unwindlib=libgcc -no-pie 2>&1 | \
// RUN:   FileCheck %s
//
// Test for header search toolchain detection.
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN:[^"]+]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/c++/4.8"
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/x86_64-linux-gnu/c++/4.8"
// CHECK: "-internal-isystem"
// CHECK: "[[TOOLCHAIN]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/c++/4.8/backward"
//
// Test for linker toolchain detection. Note that only the '-L' flags will use
// the same precise formatting of the path as the '-internal-system' flags
// above, so we just blanket wildcard match the 'crtbegin.o'.
// CHECK: "{{[^"]*}}ld{{(.exe)?}}"
// CHECK-SAME: "{{[^"]*}}/usr/lib/gcc/x86_64-linux-gnu/4.8{{/|\\\\}}crtbegin.o"
// CHECK-SAME: "-L[[TOOLCHAIN]]/usr/lib/gcc/x86_64-linux-gnu/4.8"
/// On x86_64, there is an extra usr/lib/gcc/x86_64-linux-gnu/4.8/../../../x86_64-linux-gnu but we should not test it.

/// Test we don't detect GCC installation under -B.
// RUN: %clangxx %s -### --sysroot= 2>&1 \
// RUN:   --target=aarch64-suse-linux --gcc-toolchain=%S/Inputs/opensuse_42.2_aarch64_tree/usr -no-pie | \
// RUN:   FileCheck %s --check-prefix=AARCH64
// RUN: %clangxx %s -### --sysroot= 2>&1 \
// RUN:   --target=aarch64-suse-linux -B%S/Inputs/opensuse_42.2_aarch64_tree/usr | \
// RUN:   FileCheck %s --check-prefix=NO_AARCH64

// AARCH64:        Inputs{{[^"]+}}aarch64-suse-linux/{{[^"]+}}crt1.o"
// NO_AARCH64-NOT: Inputs{{[^"]+}}aarch64-suse-linux/{{[^"]+}}crt1.o"
