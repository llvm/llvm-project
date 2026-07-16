// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -O0 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -O0 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Base {
  virtual ~Base();
  virtual int foo();
};

struct Final final : Base {
  ~Final();
  int foo() override;
};

// Devirtualized destructor call: calling a virtual destructor on a type
// that the compiler can statically resolve (e.g. a final class).
void test_devirtualized_dtor(Final *f) {
  f->~Final();
}

// CIR-LABEL: @_Z23test_devirtualized_dtorP5Final
// CIR:   cir.call @_ZN5FinalD1Ev({{.*}}) nothrow

// LLVM-LABEL: @_Z23test_devirtualized_dtorP5Final
// LLVM:   call void @_ZN5FinalD1Ev(

// OGCG-LABEL: @_Z23test_devirtualized_dtorP5Final
// OGCG:   call void @_ZN5FinalD1Ev(

// Devirtualized method call: calling a virtual method on a final class.
int test_devirtualized_method(Final *f) {
  return f->foo();
}

// CIR-LABEL: @_Z25test_devirtualized_methodP5Final
// CIR:   cir.call @_ZN5Final3fooEv(

// LLVM-LABEL: @_Z25test_devirtualized_methodP5Final
// LLVM:   call noundef i32 @_ZN5Final3fooEv(

// OGCG-LABEL: @_Z25test_devirtualized_methodP5Final
// OGCG:   call noundef i32 @_ZN5Final3fooEv(

// Covariant return type: clone() returns Final* in the derived class but
// Base* in the base. Because the return types differ, we skip devirtualization
// and go through the vtable.
struct Covariant : Base {
  virtual Covariant* clone();
};
struct FinalCovariant final : Covariant {
  FinalCovariant* clone() override;
};
Base* test_covariant_return(FinalCovariant *f) {
  return static_cast<Covariant*>(f)->clone();
}

// CIR-LABEL: @_Z21test_covariant_returnP14FinalCovariant
// CIR:   cir.vtable.get_virtual_fn_addr
// CIR:   cir.call %{{.*}}({{.*}})

// LLVM-LABEL: @_Z21test_covariant_returnP14FinalCovariant
// LLVM:   call {{.*}} %{{.*}}(

// OGCG-LABEL: @_Z21test_covariant_returnP14FinalCovariant
// OGCG:   call {{.*}} %{{.*}}(

// FinalNoOverride: final class that does not override foo().
// The compiler knows the dynamic type, so it devirtualizes to Base::foo().
struct FinalNoOverride final : Base {};
void test_final_no_override() {
  FinalNoOverride local;
  local.foo();
}

// CIR-LABEL: @_Z22test_final_no_overridev
// CIR:   cir.call @_ZN4Base3fooEv(

// LLVM-LABEL: @_Z22test_final_no_overridev
// LLVM:   call noundef i32 @_ZN4Base3fooEv(

// OGCG-LABEL: @_Z22test_final_no_overridev
// OGCG:   call noundef i32 @_ZN4Base3fooEv(

// Cross-class: method is defined in B but called through a C* (separate base).
// getCXXRecord(base)=C != devirtualizedClass=B, so devirtualization is cancelled
// and a virtual call is emitted.
struct CrossA { virtual void f(); };
struct CrossB : CrossA { void f() override; };
struct CrossC : CrossA { };
struct CrossMulti final : CrossB, CrossC {};
void test_cross_class(CrossMulti *m) { static_cast<CrossC *>(m)->f(); }

// CIR-LABEL: @_Z16test_cross_classP10CrossMulti
// CIR:   cir.vtable.get_virtual_fn_addr
// CIR:   cir.call %{{.*}}({{.*}})

// LLVM-LABEL: @_Z16test_cross_classP10CrossMulti
// LLVM:   call void %{{.*}}(

// OGCG-LABEL: @_Z16test_cross_classP10CrossMulti
// OGCG:   call void %{{.*}}(
