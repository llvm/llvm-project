// -flto causes a switch to llvm-bc object files.
// RUN: %clang -ccc-print-phases -c %s -flto 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-ACTIONS < %t %s
//
// CHECK-COMPILE-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILE-ACTIONS: 3: backend, {2}, lto-bc

// RUN: %clang -ccc-print-phases %if target={{.*-windows-msvc.*}} %{ -fuse-ld=lld %} -flto %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-ACTIONS < %t %s
//
// CHECK-COMPILELINK-ACTIONS: 0: input, "{{.*}}lto.c", c
// CHECK-COMPILELINK-ACTIONS: 1: preprocessor, {0}, cpp-output
// CHECK-COMPILELINK-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILELINK-ACTIONS: 3: backend, {2}, lto-bc
// CHECK-COMPILELINK-ACTIONS: 4: linker, {3}, image

// llvm-bc and llvm-ll outputs need to match regular suffixes
// (unfortunately).
// RUN: %clang --target=x86_64-linux-gnu %s -flto -save-temps -### 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-SUFFIXES < %t %s
//
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.i" "-x" "c" "{{.*}}lto.c"
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.bc" {{.*}}"{{.*}}lto.i"
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.o" {{.*}}"{{.*}}lto.bc"
// CHECK-COMPILELINK-SUFFIXES: "{{.*}}a.{{(out|exe)}}" {{.*}}"{{.*}}lto.o"

// RUN: %clang --target=x86_64-linux-gnu %s -flto -S -### 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-SUFFIXES < %t %s
//
// CHECK-COMPILE-SUFFIXES: "-o" "{{.*}}lto.s" "-x" "c" "{{.*}}lto.c"

// RUN: not %clang %s -emit-llvm 2>&1 | FileCheck --check-prefix=LLVM-LINK %s
// LLVM-LINK: -emit-llvm cannot be used when linking

/// With ld.bfd or gold, link against LLVMgold.
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=bfd -flto=thin -### 2>&1 | FileCheck --check-prefix=LLVMGOLD %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=gold -flto=full -### 2>&1 | FileCheck --check-prefix=LLVMGOLD %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=gold -fno-lto -flto -### 2>&1 | FileCheck --check-prefix=LLVMGOLD %s
// LLVMGOLD: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

/// lld does not need LLVMgold.
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -### 2>&1 | FileCheck --check-prefix=NO-LLVMGOLD %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=gold -flto -fno-lto -### 2>&1 | FileCheck --check-prefix=NO-LLVMGOLD %s
// NO-LLVMGOLD-NOT: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O1 -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Og -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O2 -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Os -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Oz -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O3 -### 2>&1 | FileCheck --check-prefix=O3 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Ofast -### 2>&1 | FileCheck --check-prefix=O3 %s

// O1: -plugin-opt=O1
// O2: -plugin-opt=O2
// O3: -plugin-opt=O3

// -flto passes along an explicit debugger tuning argument.
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto -glldb 2> %t
// RUN: FileCheck -check-prefix=CHECK-TUNING-LLDB < %t %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-TUNING < %t %s
//
// CHECK-TUNING-LLDB:   "-plugin-opt=-debugger-tune=lldb"
// CHECK-NO-TUNING-NOT: "-plugin-opt=-debugger-tune
//
// -flto=auto and -flto=jobserver pass along -flto=full
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=auto 2>&1 | FileCheck --check-prefix=FLTO-AUTO %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=jobserver 2>&1 | FileCheck --check-prefix=FLTO-JOBSERVER %s
//
// FLTO-AUTO: -flto=full
// FLTO-JOBSERVER: -flto=full
//

// Pass the last -flto argument.
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -flto 2>&1 | \
// RUN: FileCheck --check-prefix=FLTO-FULL %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -flto=full \
// RUN: 2>&1 | FileCheck --check-prefix=FLTO-FULL %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=full -flto=thin  \
// RUN: 2>&1 | FileCheck --check-prefix=FLTO-THIN %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto -flto=thin 2>&1 | \
// RUN: FileCheck --check-prefix=FLTO-THIN %s
//
// FLTO-FULL-NOT: -flto=thin
// FLTO-FULL: -flto=full
// FLTO-FULL-NOT: -flto=thin
//
// FLTO-THIN-NOT: -flto=full
// FLTO-THIN-NOT: "-flto"
// FLTO-THIN: -flto=thin
// FLTO-THIN-NOT: "-flto"
// FLTO-THIN-NOT: -flto=full

// -flto passes along an explicit GlobalISel flag.
// RUN: %clang --target=riscv64-linux-gnu -### %s -flto -fglobal-isel 2> %t
// RUN: FileCheck --check-prefix=CHECK-GISEL < %t %s
// RUN: %clang --target=aarch64-linux-gnu -### %s -flto -fno-global-isel 2> %t
// RUN: FileCheck --check-prefix=CHECK-DISABLE-GISEL < %t %s
//
// CHECK-GISEL:         "-plugin-opt=-global-isel=1"
// CHECK-DISABLE-GISEL: "-plugin-opt=-global-isel=0"

// -flto passes -time-passes when -ftime-report is passed
// RUN: %clang --target=x86_64-unknown-linux-gnu -### %s -flto -ftime-report 2> %t
// RUN: FileCheck --check-prefix=CHECK-TIME-REPORT < %t %s

// CHECK-TIME-REPORT: "-plugin-opt=-time-passes"
