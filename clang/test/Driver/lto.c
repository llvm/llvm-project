// -flto causes a switch to llvm-bc object files.
// RUN: %clang -ccc-print-phases -c %s -flto 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-ACTIONS < %t %s
//
// CHECK-COMPILE-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILE-ACTIONS: 3: backend, {2}, lto-bc

// RUN: %clang -ccc-print-phases %s -flto 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-ACTIONS < %t %s
//
// CHECK-COMPILELINK-ACTIONS: 0: input, "{{.*}}lto.c", c
// CHECK-COMPILELINK-ACTIONS: 1: preprocessor, {0}, cpp-output
// CHECK-COMPILELINK-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILELINK-ACTIONS: 3: backend, {2}, lto-bc
// CHECK-COMPILELINK-ACTIONS: 4: linker, {3}, image

// llvm-bc and llvm-ll outputs need to match regular suffixes
// (unfortunately).
// RUN: %clang %s -flto -save-temps -### 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-SUFFIXES < %t %s
//
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.i" "-x" "c" "{{.*}}lto.c"
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.bc" {{.*}}"{{.*}}lto.i"
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.o" {{.*}}"{{.*}}lto.bc"
// CHECK-COMPILELINK-SUFFIXES: "{{.*}}a.{{(out|exe)}}" {{.*}}"{{.*}}lto.o"

// RUN: %clang %s -flto -S -### 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-SUFFIXES < %t %s
//
// CHECK-COMPILE-SUFFIXES: "-o" "{{.*}}lto.s" "-x" "c" "{{.*}}lto.c"

// RUN: not %clang %s -emit-llvm 2>&1 | FileCheck --check-prefix=LLVM-LINK %s
// LLVM-LINK: -emit-llvm cannot be used when linking

// -flto should cause link using gold plugin
// RUN: %clang -target x86_64-unknown-linux -### %s -flto 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-LTO-ACTION < %t %s
//
// CHECK-LINK-LTO-ACTION: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

// -flto=full should cause link using gold plugin
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=full 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-FULL-ACTION < %t %s
//
// CHECK-LINK-FULL-ACTION: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

// Check that subsequent -fno-lto takes precedence
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=full -fno-lto 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-NOLTO-ACTION < %t %s
//
// CHECK-LINK-NOLTO-ACTION-NOT: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

// -flto passes along an explicit debugger tuning argument.
// RUN: %clang -target x86_64-unknown-linux -### %s -flto -glldb 2> %t
// RUN: FileCheck -check-prefix=CHECK-TUNING-LLDB < %t %s
// RUN: %clang -target x86_64-unknown-linux -### %s -flto -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-TUNING < %t %s
//
// CHECK-TUNING-LLDB:   "-plugin-opt=-debugger-tune=lldb"
// CHECK-NO-TUNING-NOT: "-plugin-opt=-debugger-tune
