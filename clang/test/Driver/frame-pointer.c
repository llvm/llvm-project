// RUN: %clang --target=i386-pc-linux -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=i386-pc-linux -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=i386-pc-linux -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang --target=i386-pc-linux -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang --target=i386-pc-linux -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s

// RUN: %clang --target=i386-linux-android -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s
// RUN: %clang --target=i386-linux-android -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s
// RUN: %clang --target=i386-linux-android -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s

// RUN: %clang --target=x86_64-pc-linux -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=x86_64-pc-linux -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang --target=x86_64-pc-linux -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang --target=x86_64-pc-linux -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang --target=x86_64-pc-linux -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s
// RUN: %clang --target=x86_64-pc-win32-macho -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-MACHO-64 %s

// RUN: %clang --target=x86_64-linux-android -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s
// RUN: %clang --target=x86_64-linux-android -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s
// RUN: %clang --target=x86_64-linux-android -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s

// Trust the above to get the optimizations right, and just test other targets
// that want this by default.
// RUN: %clang --target=s390x-pc-linux -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=s390x-pc-linux -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s

// RUN: %clang --target=powerpc-unknown-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=powerpc-unknown-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang --target=powerpc64-unknown-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=powerpc64-unknown-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang --target=powerpc64le-unknown-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=powerpc64le-unknown-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s

// RUN: %clang --target=mips-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=mips-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=mipsel-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=mipsel-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=mips64-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=mips64-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=mips64el-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=mips64el-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s

// RUN: %clang --target=riscv32-unknown-elf -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=riscv32-unknown-elf -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=riscv32-unknown-elf -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang --target=riscv32-unknown-elf -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang --target=riscv32-unknown-elf -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s

// RUN: %clang --target=riscv64-unknown-elf -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=riscv64-unknown-elf -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang --target=riscv64-unknown-elf -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang --target=riscv64-unknown-elf -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang --target=riscv64-unknown-elf -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s

// RUN: %clang --target=riscv32-unknown-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=riscv32-unknown-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=riscv32-unknown-linux-gnu -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang --target=riscv32-unknown-linux-gnu -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang --target=riscv32-unknown-linux-gnu -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s

// RUN: %clang --target=riscv64-unknown-linux-gnu -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s

// RUN: %clang --target=riscv64-linux-android -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s
// RUN: %clang --target=riscv64-linux-android -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s
// RUN: %clang --target=riscv64-linux-android -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-ANDROID %s

// RUN: %clang --target=loongarch32 -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang --target=loongarch32 -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang --target=loongarch32 -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang --target=loongarch32 -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang --target=loongarch32 -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s

// RUN: %clang --target=loongarch64 -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang --target=loongarch64 -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang --target=loongarch64 -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang --target=loongarch64 -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang --target=loongarch64 -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s

// CHECK0-32: -mframe-pointer=all
// CHECK1-32-NOT: -mframe-pointer=all
// CHECK2-32-NOT: -mframe-pointer=all
// CHECK3-32-NOT: -mframe-pointer=all
// CHECKs-32-NOT: -mframe-pointer=all

// CHECK0-64: -mframe-pointer=all
// CHECK1-64-NOT: -mframe-pointer=all
// CHECK2-64-NOT: -mframe-pointer=all
// CHECK3-64-NOT: -mframe-pointer=all
// CHECKs-64-NOT: -mframe-pointer=all
// CHECK-MACHO-64: -mframe-pointer=all

// CHECK-ANDROID: -mframe-pointer=non-leaf
