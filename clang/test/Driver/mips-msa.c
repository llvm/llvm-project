// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:   -Wa,-mmsa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-MMSA
// CHECK-MMSA: "-cc1" {{.*}}"-mmsa"

// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:   -Wa,-mmsa,-mno-msa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-NOMMSA
// CHECK-NOMMSA:     "-cc1"
// CHECK-NOMMSA-NOT: "-mssa"

// RUN: not %clang -### -c --target=x86_64 -Wa,-mmsa -Wa,-mno-msa %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: error: unsupported argument '-mmsa' to option '-Wa,'
// ERR: error: unsupported argument '-mno-msa' to option '-Wa,'
