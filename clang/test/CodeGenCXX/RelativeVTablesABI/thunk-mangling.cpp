// Check that virtual thunks are unaffected by the relative ABI.
// The offset of thunks is mangled into the symbol name, which could result in
// linking errors for binaries that want to look for symbols in SOs made with
// this ABI.
// Running that linked binary still won't work since we're using conflicting
// ABIs, but we should still be able to link.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// This would be normally n24 (3 ptr widths) but is 12 since the vtable is
// entierely made of i32s now.
// CHECK: _ZTv0_n12_N7Derived1fEi

class Base {
public:
  virtual int f(int x);

private:
  long x;
};

class Derived : public virtual Base {
public:
  virtual int f(int x);

private:
  long y;
};

int Base::f(int x) { return x + 1; }
int Derived::f(int x) { return x + 2; }
