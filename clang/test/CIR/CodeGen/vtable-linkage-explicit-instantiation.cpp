// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

template <typename T>
struct Base {
  virtual ~Base() {}
  virtual void foo() {}
  T val;
};

extern template class Base<int>;

void use(Base<int> *p) {
  p->foo();
}

// Constructing a Base<int> forces the vtable to be emitted.
Base<int> make_base() {
  Base<int> b;
  return b;
}

// Class with a key function (out-of-line virtual destructor).
template <typename T>
struct KeyBase {
  virtual ~KeyBase();
  T val;
};

template <typename T> KeyBase<T>::~KeyBase() {}

extern template class KeyBase<int>;

void use_key() {
  KeyBase<int> k;
  (void)k;
}

// Base has no key function and extern template declaration, so the vtable
// gets external linkage.
// CHECK: cir.global "private" external @_ZTV4BaseIiE

// The key function (~KeyBase) is not instantiated in this TU, so the vtable
// also gets external linkage.
// CHECK: cir.global "private" external @_ZTV7KeyBaseIiE

// Verify the virtual call goes through the vtable.
// CHECK: cir.func {{.*}} @_Z3useP4BaseIiE
// CHECK:   cir.vtable.get_vptr
// CHECK:   cir.vtable.get_virtual_fn_addr

// LLVM: @_ZTV4BaseIiE = external global
// LLVM: @_ZTV7KeyBaseIiE = external global
// LLVM: define {{.*}} @_Z3useP4BaseIiE

// OGCG: @_ZTV4BaseIiE = external unnamed_addr constant
// OGCG: @_ZTV7KeyBaseIiE = external unnamed_addr constant
// OGCG: define {{.*}} @_Z3useP4BaseIiE
