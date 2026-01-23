// RUN: %clang --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=x86_64-apple-darwin10 -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=x86_64-pc-win32 -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOMFCALL
// RUN: %clang --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -fsanitize-cfi-cross-dso -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOMFCALL
// RUN: %clang --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi-derived-cast -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-DCAST
// RUN: %clang --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi-unrelated-cast -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-UCAST
// RUN: %clang --target=x86_64-linux-gnu -flto -fvisibility=hidden -fsanitize=cfi-nvcall -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NVCALL
// RUN: %clang --target=x86_64-linux-gnu -flto -fvisibility=hidden -fsanitize=cfi-vcall -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-VCALL
// RUN: %clang --target=arm-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=aarch64-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=arm-linux-android -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=arm-none-eabi -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=thumb-none-eabi -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=aarch64-linux-android -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=aarch64_be -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=riscv32 -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=riscv64 -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang --target=loongarch64 -fvisibility=hidden -fsanitize=cfi -flto -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI
// CHECK-CFI: -emit-llvm-bc{{.*}}-fsanitize=cfi-derived-cast,cfi-icall,cfi-mfcall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall
// CHECK-CFI-NOMFCALL: -emit-llvm-bc{{.*}}-fsanitize=cfi-derived-cast,cfi-icall,cfi-unrelated-cast,cfi-nvcall,cfi-vcall
// CHECK-CFI-DCAST: -emit-llvm-bc{{.*}}-fsanitize=cfi-derived-cast
// CHECK-CFI-UCAST: -emit-llvm-bc{{.*}}-fsanitize=cfi-unrelated-cast
// CHECK-CFI-NVCALL: -emit-llvm-bc{{.*}}-fsanitize=cfi-nvcall
// CHECK-CFI-VCALL: -emit-llvm-bc{{.*}}-fsanitize=cfi-vcall

// RUN: not %clang --target=x86_64-linux-gnu -fvisibility=hidden -flto -fsanitize=cfi-derived-cast -fno-lto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOLTO
// CHECK-CFI-NOLTO: '-fsanitize=cfi-derived-cast' only allowed with '-flto'

// RUN: not %clang --target=x86_64-linux-gnu -flto -fsanitize=cfi-derived-cast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOVIS
// CHECK-CFI-NOVIS: error: invalid argument '-fsanitize=cfi-derived-cast' only allowed with '-fvisibility='

// RUN: %clang --target=x86_64-pc-win32 -flto -fsanitize=cfi-derived-cast -resource-dir=%S/Inputs/resource_dir -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOVIS-NOERROR
// RUN: echo > %t.o
// RUN: %clang --target=x86_64-linux-gnu -flto -fsanitize=cfi-derived-cast -resource-dir=%S/Inputs/resource_dir %t.o -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOVIS-NOERROR
// CHECK-CFI-NOVIS-NOERROR-NOT: only allowed with

// RUN: not %clang --target=mips-unknown-linux -fsanitize=cfi-icall %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-ICALL-MIPS
// CHECK-CFI-ICALL-MIPS: error: unsupported option '-fsanitize=cfi-icall' for target 'mips-unknown-linux'

// RUN: not %clang --target=x86_64-apple-darwin10 -mmacos-version-min=10.7 -flto -fsanitize=cfi-vcall -fno-sanitize-trap=cfi -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-OLD-MACOS
// CHECK-CFI-NOTRAP-OLD-MACOS: error: unsupported option '-fno-sanitize-trap=cfi-vcall' for target 'x86_64-apple-darwin10'

// RUN: %clang --target=x86_64-pc-win32 -flto -fsanitize=cfi-vcall -fno-sanitize-trap=cfi -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NOTRAP-WIN
// CHECK-CFI-NOTRAP-WIN: -emit-llvm-bc
// CHECK-CFI-NOTRAP-WIN-NOT: -fsanitize-trap=cfi

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -fsanitize-cfi-cross-dso -flto -fvisibility=hidden -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-CROSS-DSO
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -flto -fvisibility=hidden -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NO-CROSS-DSO
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -fsanitize-cfi-cross-dso -fno-sanitize-cfi-cross-dso -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-NO-CROSS-DSO
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -fno-sanitize-cfi-cross-dso -fsanitize-cfi-cross-dso -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-CROSS-DSO
// CHECK-CFI-CROSS-DSO: -emit-llvm-bc
// CHECK-CFI-CROSS-DSO: -fsanitize-cfi-cross-dso
// CHECK-CFI-NO-CROSS-DSO: -emit-llvm-bc
// CHECK-CFI-NO-CROSS-DSO-NOT: -fsanitize-cfi-cross-dso

// RUN: not %clang --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi-mfcall -fsanitize-cfi-cross-dso -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-MFCALL-CROSS-DSO
// CHECK-CFI-MFCALL-CROSS-DSO: '-fsanitize=cfi-mfcall' not allowed with '-fsanitize-cfi-cross-dso'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi-icall -fsanitize-cfi-icall-generalize-pointers -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-GENERALIZE-POINTERS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi-icall -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-CFI-GENERALIZE-POINTERS
// CHECK-CFI-GENERALIZE-POINTERS: -fsanitize-cfi-icall-generalize-pointers
// CHECK-NO-CFI-GENERALIZE-POINTERS-NOT: -fsanitize-cfi-icall-generalize-pointers

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=cfi-icall -fsanitize-cfi-icall-generalize-pointers -fsanitize-cfi-cross-dso -fvisibility=hidden -flto -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-GENERALIZE-AND-CROSS-DSO
// CHECK-CFI-GENERALIZE-AND-CROSS-DSO: error: invalid argument '-fsanitize-cfi-cross-dso' not allowed with '-fsanitize-cfi-icall-generalize-pointers'

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize-cfi-icall-generalize-pointers -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KCFI-GENERALIZE-POINTERS
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-KCFI-GENERALIZE-POINTERS
// CHECK-KCFI-GENERALIZE-POINTERS: -fsanitize-cfi-icall-generalize-pointers
// CHECK-NO-KCFI-GENERALIZE-POINTERS-NOT: -fsanitize-cfi-icall-generalize-pointers

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi-icall -fsanitize-cfi-canonical-jump-tables -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-CANONICAL-JUMP-TABLES
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi-icall -fno-sanitize-cfi-canonical-jump-tables -fvisibility=hidden -flto -c %s -resource-dir=%S/Inputs/resource_dir -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-CFI-CANONICAL-JUMP-TABLES
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi-icall -fvisibility=hidden -flto -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-CANONICAL-JUMP-TABLES
// CHECK-CFI-CANONICAL-JUMP-TABLES: -fsanitize-cfi-canonical-jump-tables
// CHECK-NO-CFI-CANONICAL-JUMP-TABLES-NOT: -fsanitize-cfi-canonical-jump-tables

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=cfi -fsanitize-stats -flto -fvisibility=hidden -c -resource-dir=%S/Inputs/resource_dir %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CFI-STATS
// CHECK-CFI-STATS: -fsanitize-stats

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize=cfi -flto -fvisibility=hidden %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KCFI-NOCFI
// CHECK-KCFI-NOCFI: error: invalid argument '-fsanitize=kcfi' not allowed with '-fsanitize=cfi'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fsanitize-trap=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KCFI-NOTRAP
// CHECK-KCFI-NOTRAP: error: unsupported argument 'kcfi' to option '-fsanitize-trap='

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KCFI
// CHECK-KCFI: "-fsanitize=kcfi"

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kcfi -fno-sanitize-recover=kcfi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KCFI-RECOVER
// CHECK-KCFI-RECOVER: error: unsupported argument 'kcfi' to option '-fno-sanitize-recover='

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=kcfi,function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-KCFI-FUNCTION
// CHECK-KCFI-FUNCTION: error: invalid argument '-fsanitize=kcfi' not allowed with '-fsanitize=function'
