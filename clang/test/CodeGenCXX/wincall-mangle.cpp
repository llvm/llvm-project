// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-gnu -o - -emit-llvm %s | FileCheck --check-prefixes=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64apx-pc-windows-cygnus -o - -emit-llvm %s | FileCheck --check-prefixes=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-uefi -o - -emit-llvm %s | FileCheck --check-prefixes=MSVC %s

// wincall is the default calling convention for x86_64apx PE/COFF targets
// (Windows, Cygwin, MSYS and UEFI).
// C++ symbols get a @win suffix (both Itanium and MSVC-style manglings) so
// that the linker can catch calling-convention mismatches, and function
// pointer types carry the wincall vendor qualifier in the Itanium mangling.

typedef void(__attribute__((wincall)) *W)(int);

struct C {
  void __attribute__((wincall)) m(int);
};

void C::m(int a) {
  // ITANIUM-LABEL: define dso_local x86_wincallcc void @"\01_ZN1C1mEi@win"
  // MSVC-LABEL: define dso_local x86_wincallcc void @"?m@C@@QEAAXH@Z@win"
  (void)a;
}

void f(int);

void g() {
  // ITANIUM-LABEL: define dso_local x86_wincallcc void @"\01_Z1gv@win"
  // MSVC-LABEL: define dso_local x86_wincallcc void @"?g@@YAXXZ@win"
  f(1);
  // ITANIUM: call x86_wincallcc void @"\01_Z1fi@win"
  // MSVC: call x86_wincallcc void @"?f@@YAXH@Z@win"
}

// Function pointer types get a U7wincall vendor qualifier (Itanium) or are
// modelled as a distinct type in the MSVC mangling.
template <typename T> T func_as_int(T x);
W w;
W test() {
  // ITANIUM-LABEL: define dso_local x86_wincallcc noundef ptr @"\01_Z4testv@win"
  // ITANIUM: call x86_wincallcc noundef ptr @"\01_Z11func_as_intIPU7wincallFviEET_S2_@win"
  // MSVC-LABEL: define dso_local x86_wincallcc noundef ptr @"?test@@YAP6AXH@ZXZ@win"
  return func_as_int<W>(w);
}
