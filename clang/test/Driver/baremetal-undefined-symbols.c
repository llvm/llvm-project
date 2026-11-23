// Check the arguments are correctly passed

// Make sure -T is the last with gcc-toolchain option
// RUN: %clang -### --target=aarch64-none-elf --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LD %s
// RUN: %clang -### --target=armv6m-none-eabi --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LD %s
// RUN: %clang -### --target=riscv32 --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LD %s
// CHECK-LD: {{.*}} "-T" "a.lds" "-u" "foo" {{.*}} "--defsym=FOO=10"
