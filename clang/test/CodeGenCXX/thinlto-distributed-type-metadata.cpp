// Test distributed ThinLTO backend handling of type tests

// REQUIRES: x86-registered-target

// Ensure that a distributed backend invocation of ThinLTO lowers the type test
// as expected.
// RUN: %clang_cc1 -flto=thin -flto-unit -triple x86_64-unknown-linux -fwhole-program-vtables -emit-llvm-bc -o %t.o %s
// RUN: llvm-dis %t.o -o - | FileCheck --check-prefix=TT %s
// RUN: llvm-lto -thinlto -o %t2 %t.o
// RUN: %clang -target x86_64-unknown-linux -O2 -o %t3.o -x ir %t.o -c -fthinlto-index=%t2.thinlto.bc -save-temps=obj
// RUN: llvm-dis %t.s.4.opt.bc -o - | FileCheck --check-prefix=OPT %s
// llvm-nm %t3.o | FileCheck --check-prefix=NM %s

// The pre-link bitcode produced by clang should contain a type test assume
// sequence.
// TT: [[TTREG:%[0-9]+]] = call i1 @llvm.public.type.test({{.*}}, metadata !"_ZTS1A")
// TT: void @llvm.assume(i1 [[TTREG]])

// The ThinLTO backend optimized bitcode should not have any type tests.
// OPT-NOT: @llvm.type.test
// OPT-NOT: @llvm.public.type.test
// We should have only one @llvm.assume call, the one that was expanded
// from the builtin in the IR below, not the one fed by the type test.
// OPT: %cmp = icmp ne ptr %{{.*}}, null
// OPT: void @llvm.assume(i1 %cmp)
// Check after the builtin assume again that we don't have any type tests
// OPT-NOT: @llvm.type.test
// OPT-NOT: @llvm.public.type.test

// NM: T _Z2afP1A

// Also check type test are lowered when the distributed ThinLTO backend clang
// invocation is passed an empty index file, in which case a non-ThinLTO
// compilation pipeline is invoked. If not lowered then LLVM CodeGen may assert.
// RUN: touch %t4.thinlto.bc
// O2 new PM
// RUN: %clang -target x86_64-unknown-linux -O2 -o %t4.o -x ir %t.o -c -fthinlto-index=%t4.thinlto.bc -save-temps=obj
// RUN: llvm-dis %t.s.4.opt.bc -o - | FileCheck --check-prefix=OPT %s
// llvm-nm %t4.o | FileCheck --check-prefix=NM %s
// O0 new PM
// RUN: %clang -target x86_64-unknown-linux -O0 -o %t4.o -x ir %t.o -c -fthinlto-index=%t4.thinlto.bc -save-temps=obj
// RUN: llvm-dis %t.s.4.opt.bc -o - | FileCheck --check-prefix=OPT %s
// llvm-nm %t4.o | FileCheck --check-prefix=NM %s

struct A {
  A();
  virtual void f();
};

struct B : virtual A {
  B();
};

A::A() {}
B::B() {}

void A::f() {
}

void af(A *a) {
  __builtin_assume(a != 0);
  a->f();
}
