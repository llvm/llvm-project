// -flto=thin causes a switch to llvm-bc object files.
// RUN: %clang -ccc-print-phases -c %s -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-ACTIONS < %t %s
//
// CHECK-COMPILE-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILE-ACTIONS: 3: backend, {2}, lto-bc

// RUN: %clang -ccc-print-phases %s -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-ACTIONS < %t %s
//
// CHECK-COMPILELINK-ACTIONS: 0: input, "{{.*}}thinlto.c", c
// CHECK-COMPILELINK-ACTIONS: 1: preprocessor, {0}, cpp-output
// CHECK-COMPILELINK-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILELINK-ACTIONS: 3: backend, {2}, lto-bc
// CHECK-COMPILELINK-ACTIONS: 4: linker, {3}, image

// -flto=thin should cause link using gold plugin with thinlto option,
// also confirm that it takes precedence over earlier -fno-lto and -flto=full.
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=full -fno-lto -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-ACTION < %t %s
//
// CHECK-LINK-THIN-ACTION: "-plugin" "{{.*}}/LLVMgold.so"
// CHECK-LINK-THIN-ACTION: "-plugin-opt=thinlto"

// Check that subsequent -flto=full takes precedence
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -flto=full 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-FULL-ACTION < %t %s
//
// CHECK-LINK-FULL-ACTION: "-plugin" "{{.*}}/LLVMgold.so"
// CHECK-LINK-FULL-ACTION-NOT: "-plugin-opt=thinlto"

// Check that subsequent -fno-lto takes precedence
// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -fno-lto 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-NOLTO-ACTION < %t %s
//
// CHECK-LINK-NOLTO-ACTION-NOT: "-plugin" "{{.*}}/LLVMgold.so"
// CHECK-LINK-NOLTO-ACTION-NOT: "-plugin-opt=thinlto"
