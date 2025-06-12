// Check the arguments are correctly passed

// Make sure -T is the last with gcc-toolchain option
// RUN: %clang -### --target=riscv32 --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LD %s
// CHECK-LD: {{.*}} "--defsym=FOO=10" {{.*}} "-u" "foo" {{.*}} "-T" "a.lds"

// TODO: Merge this test with the above in the last patch when finally integrating riscv
// Make sure -T is the last with gcc-toolchain option
// RUN: %clang -### --target=aarch64-none-elf --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-LD %s
// RUN: %clang -### --target=armv6m-none-eabi --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-LD %s
// CHECK-ARM-LD: {{.*}} "-T" "a.lds" "-u" "foo" {{.*}} "--defsym=FOO=10"
