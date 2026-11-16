// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fno-finite-loops -emit-llvm %s -o %t-nomp.ll
// RUN: FileCheck --input-file=%t-nomp.ll %s -check-prefix=LLVM-NOMP

void foo() {}

// CIR: cir.func {{.*}}@_Z3foov{{.*}}attributes {{.*}}mustprogress

// LLVM: define {{.*}} void @_Z3foov(){{.*}} #[[LATTR:[0-9]+]]

// With -fno-finite-loops, checkIfFunctionMustProgress() is false, so no
// function should carry 'mustprogress' at all.
// LLVM-NOMP: define {{.*}} void @_Z3foov()
// LLVM-NOMP-NOT: mustprogress

// Trivial while removes the mustprogress.
void trivial_infinite_while() {
  while (true) {
  }
}

// CIR: cir.func {{.*}}@_Z22trivial_infinite_whilev
// CIR-NOT: mustprogress

// LLVM: define {{.*}} void @_Z22trivial_infinite_whilev(){{.*}} #[[NOMPATTR:[0-9]+]]

// As does trivial for.
void trivial_infinite_for() {
  for (;;) {
  }
}

// CIR: cir.func {{.*}}@_Z20trivial_infinite_forv
// CIR-NOT: mustprogress

// LLVM: define {{.*}} void @_Z20trivial_infinite_forv(){{.*}} #[[NOMPATTR]]

// And trivial do.
void trivial_infinite_do() {
  do {
  } while (true);
}

// CIR: cir.func {{.*}}@_Z19trivial_infinite_dov
// CIR-NOT: mustprogress

// LLVM: define {{.*}} void @_Z19trivial_infinite_dov(){{.*}} #[[NOMPATTR]]

// But not if there is a side effect!
void non_trivial_infinite_loop() {
  while (true) {
    asm volatile("");
  }
}

// CIR: cir.func {{.*}}@_Z25non_trivial_infinite_loopv{{.*}}attributes {{.*}}mustprogress

// LLVM: define {{.*}} void @_Z25non_trivial_infinite_loopv(){{.*}} #[[LATTR]]

// LLVM: attributes #[[LATTR]] = {{[{].*}}mustprogress{{.*}}}
// LLVM-NOT: attributes #[[NOMPATTR]] = {{[{].*}}mustprogress{{.*}}}
// LLVM: attributes #[[NOMPATTR]]
