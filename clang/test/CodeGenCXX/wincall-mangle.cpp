// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-gnu -o - -emit-llvm %s | FileCheck %s

// wincall is the default calling convention for x86_64apx-windows targets.
// C++ symbols get a @win suffix (both Itanium and MSVC-style manglings) so
// that the linker can catch calling-convention mismatches, and function
// pointer types carry the wincall vendor qualifier in the Itanium mangling.

typedef void(__attribute__((wincall)) *W)(int);

struct C {
  void __attribute__((wincall)) m(int);
};

void C::m(int a) {
  // CHECK-LABEL: define dso_local x86_wincallcc void @"\01_ZN1C1mEi@win"
  (void)a;
}

void f(int);

void g() {
  // CHECK-LABEL: define dso_local x86_wincallcc void @"\01_Z1gv@win"
  f(1);
  // CHECK: call x86_wincallcc void @"\01_Z1fi@win"
}

// Function pointer types get a U7wincall vendor qualifier.
template <typename T> T func_as_int(T x);
W w;
W test() {
  // CHECK-LABEL: define dso_local x86_wincallcc noundef ptr @"\01_Z4testv@win"
  // CHECK: call x86_wincallcc noundef ptr @"\01_Z11func_as_intIPU7wincallFviEET_S2_@win"
  return func_as_int<W>(w);
}
