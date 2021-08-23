// RUN: %clang_cc1 -triple i686-windows-msvc -emit-llvm -std=c++14 \
// RUN:    -fno-threadsafe-statics -fms-extensions -O1 -mconstructor-aliases \
// RUN:    -disable-llvm-passes -o - %s -w -fms-compatibility-version=19.00 | \
// RUN:    FileCheck %s

struct HasDtor {
  ~HasDtor();
  int o;
};
struct HasImplicitDtor1 {
  HasDtor o;
};
struct __declspec(dllexport) CtorClosureOuter {
  struct __declspec(dllexport) CtorClosureInner {
    CtorClosureInner(const HasImplicitDtor1 &v = {}) {}
  };
};

// CHECK-LABEL: $"??1HasImplicitDtor1@@QAE@XZ" = comdat any
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FCtorClosureInner@CtorClosureOuter@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
