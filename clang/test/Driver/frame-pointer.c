// RUN: %clang -target i386-pc-linux -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target i386-pc-linux -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang -target i386-pc-linux -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang -target i386-pc-linux -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang -target i386-pc-linux -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s


// RUN: %clang -target x86_64-pc-linux -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target x86_64-pc-linux -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang -target x86_64-pc-linux -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang -target x86_64-pc-linux -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang -target x86_64-pc-linux -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s
// RUN: %clang -target x86_64-pc-win32-macho -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-MACHO-64 %s

// Trust the above to get the optimizations right, and just test other targets
// that want this by default.
// RUN: %clang -target s390x-pc-linux -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target s390x-pc-linux -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s

// RUN: %clang -target powerpc-unknown-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target powerpc-unknown-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s

// RUN: %clang -target mips-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target mips-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang -target mipsel-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target mipsel-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang -target mips64-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target mips64-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang -target mips64el-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target mips64el-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s

// RUN: %clang -target riscv32-unknown-elf -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target riscv32-unknown-elf -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang -target riscv32-unknown-elf -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang -target riscv32-unknown-elf -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang -target riscv32-unknown-elf -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s

// RUN: %clang -target riscv64-unknown-elf -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target riscv64-unknown-elf -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang -target riscv64-unknown-elf -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang -target riscv64-unknown-elf -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang -target riscv64-unknown-elf -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s

// RUN: %clang -target riscv32-unknown-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-32 %s
// RUN: %clang -target riscv32-unknown-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-32 %s
// RUN: %clang -target riscv32-unknown-linux-gnu -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-32 %s
// RUN: %clang -target riscv32-unknown-linux-gnu -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-32 %s
// RUN: %clang -target riscv32-unknown-linux-gnu -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-32 %s

// RUN: %clang -target riscv64-unknown-linux-gnu -### -S -O0 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK0-64 %s
// RUN: %clang -target riscv64-unknown-linux-gnu -### -S -O1 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK1-64 %s
// RUN: %clang -target riscv64-unknown-linux-gnu -### -S -O2 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK2-64 %s
// RUN: %clang -target riscv64-unknown-linux-gnu -### -S -O3 %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK3-64 %s
// RUN: %clang -target riscv64-unknown-linux-gnu -### -S -Os %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECKs-64 %s

// CHECK0-32: -mdisable-fp-elim
// CHECK1-32-NOT: -mdisable-fp-elim
// CHECK2-32-NOT: -mdisable-fp-elim
// CHECK3-32-NOT: -mdisable-fp-elim
// CHECKs-32-NOT: -mdisable-fp-elim

// CHECK0-64: -mdisable-fp-elim
// CHECK1-64-NOT: -mdisable-fp-elim
// CHECK2-64-NOT: -mdisable-fp-elim
// CHECK3-64-NOT: -mdisable-fp-elim
// CHECKs-64-NOT: -mdisable-fp-elim
// CHECK-MACHO-64: -mdisable-fp-elim
