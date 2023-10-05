// REQUIRES: system-darwin
// RUN: rm -rf %t_devirt && mkdir %t_devirt
// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir

// RUN: %clang -flto=thin -target arm64-apple-ios16 -emit-llvm -c -fwhole-program-vtables -fvirtual-function-elimination -mllvm -enable-vfe-summary -o %t.main.bc %t.dir/vfe-main.cpp
// RUN: llvm-dis %t.main.bc -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang -flto=thin -target arm64-apple-ios16 -emit-llvm -c -fwhole-program-vtables -fvirtual-function-elimination -mllvm -enable-vfe-summary -o %t.input.bc %t.dir/vfe-input.cpp
// RUN: llvm-dis %t.input.bc -o - | FileCheck %s --check-prefix=INPUT

// RUN: llvm-lto2 run %t.main.bc %t.input.bc -save-temps -enable-vfe-on-thinlto -o %t.out \
// RUN:   -r=%t.main.bc,__Z6test_1P1A,pl \
// RUN:   -r=%t.main.bc,__Z4testv,pl \
// RUN:   -r=%t.main.bc,__Znwm, \
// RUN:   -r=%t.main.bc,__ZN1AC1Ev,pl \
// RUN:   -r=%t.main.bc,___gxx_personality_v0, \
// RUN:   -r=%t.main.bc,__ZN1AC2Ev,pl \
// RUN:   -r=%t.main.bc,__ZN1BC2Ev,pl \
// RUN:   -r=%t.main.bc,__ZN1BC1Ev,pl \
// RUN:   -r=%t.main.bc,__ZN1A3fooEv,pl \
// RUN:   -r=%t.main.bc,__ZN1A3barEv,pl \
// RUN:   -r=%t.main.bc,__Z6test_2P1B,pl \
// RUN:   -r=%t.main.bc,_main,plx \
// RUN:   -r=%t.main.bc,__ZdlPv \
// RUN:   -r=%t.main.bc,__ZTV1A,pl \
// RUN:   -r=%t.main.bc,__ZTV1B, \
// RUN:   -r=%t.main.bc,__ZTVN10__cxxabiv117__class_type_infoE, \
// RUN:   -r=%t.main.bc,__ZTS1A,pl \
// RUN:   -r=%t.main.bc,__ZTI1A,pl \
// RUN:   -r=%t.input.bc,__ZN1CC2Ev,pl \
// RUN:   -r=%t.input.bc,__ZN1CC1Ev,pl \
// RUN:   -r=%t.input.bc,__Z6test_3P1C,pl \
// RUN:   -r=%t.input.bc,__Z6test_4P1C,pl \
// RUN:   -r=%t.input.bc,__Z6test_5P1CMS_FvvE,pl \
// RUN:   -r=%t.input.bc,__ZTV1C,

// RUN: llvm-dis %t.out.1.0.preopt.bc -o - | FileCheck %s --check-prefix=ORIGINAL
// RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=AFTERVFE
// ORIGINAL: define{{.*}}_ZN1A3barEv
// AFTERVFE-NOT: define{{.*}}_ZN1A3barEv

//--- vfe-main.cpp
// CHECK: @_ZTV1A = {{.*}}constant {{.*}}@_ZTI1A{{.*}}@_ZN1A3fooEv{{.*}}_ZN1A3barEv{{.*}}!type [[A16:![0-9]+]]
// CHECK-NOT: @_ZTV1B = {{.*}}!type
struct __attribute__((visibility("hidden"))) A {
  A();
  virtual void foo();
  virtual void bar();
};

__attribute__((used)) void test_1(A *p) {
  // CHECK-LABEL: define{{.*}} void @_Z6test_1P1A
  // CHECK: [[VTABLE:%.+]] = load
  // CHECK: @llvm.type.checked.load(ptr {{%.+}}, i32 0, metadata !"_ZTS1A")
  p->foo();
}

__attribute__((used)) A *test() {
  return new (A)();
}

struct __attribute__((visibility("hidden"))) B {
  B();
  virtual void foo();
};

A::A() {}
B::B() {}
void A::foo() {}
void A::bar() {}
void test_2(B *p) {
  // CHECK-LABEL: define{{.*}} void @_Z6test_2P1B
  // CHECK: [[VTABLE:%.+]] = load
  // CHECK: @llvm.type.checked.load(ptr {{%.+}}, i32 0, metadata !"_ZTS1B")
  p->foo();
}

// INPUT-NOT: @_ZTV1C = {{.*}}!type
// INPUT-LABEL: define{{.*}} void @_Z6test_3P1C
// INPUT: [[LOAD:%.+]] = {{.*}}call { ptr, i1 } @llvm.type.checked.load(ptr {{%.+}}, i32 0, metadata !"_ZTS1C")
// INPUT: [[FN_PTR:%.+]] = extractvalue { ptr, i1 } [[LOAD]], 0
// INPUT: call void [[FN_PTR]](

// INPUT-LABEL: define{{.*}} void @_Z6test_4P1C
// INPUT: [[LOAD:%.+]] = {{.*}}call { ptr, i1 } @llvm.type.checked.load(ptr {{%.+}}, i32 8, metadata !"_ZTS1C")
// INPUT: [[FN_PTR:%.+]] = extractvalue { ptr, i1 } [[LOAD]], 0
// INPUT: call void [[FN_PTR]](

int main() {
}

// CHECK: [[BAR:\^[0-9]+]] = gv: (name: "_ZN1A3barEv", {{.*}}
// CHECK: FuncsWithNonVtableRef
// CHECK-NOT: [[BAR]]
//--- vfe-input.cpp
struct __attribute__((visibility("hidden"))) C {
  C();
  virtual void foo();
  virtual void bar();
};

C::C() {}
void test_3(C *p) {
  // C has hidden visibility, so we generate type.checked.load to allow VFE.
  p->foo();
}

void test_4(C *p) {
  // When using type.checked.load, we pass the vtable offset to the intrinsic,
  // rather than adding it to the pointer with a GEP.
  p->bar();
}

void test_5(C *p, void (C::*q)(void)) {
  // We also use type.checked.load for the virtual side of member function
  // pointer calls. We use a GEP to calculate the address to load from and pass
  // 0 as the offset to the intrinsic, because we know that the load must be
  // from exactly the point marked by one of the function-type metadatas (in
  // this case "_ZTSM1CFvvE.virtual"). If we passed the offset from the member
  // function pointer to the intrinsic, this information would be lost. No
  // codegen changes on the non-virtual side.
  (p->*q)();
}
