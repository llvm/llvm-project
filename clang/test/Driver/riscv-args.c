// Check the arguments are correctly passed

// Make sure -T is the last with gcc-toolchain option
// RUN: %clang -### -target riscv32 --gcc-toolchain= -Xlinker --defsym=FOO=10 -T a.lds -u foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LD %s
// CHECK-LD: {{.*}} "--defsym=FOO=10" {{.*}} "-u" "foo" {{.*}} "-T" "a.lds"
