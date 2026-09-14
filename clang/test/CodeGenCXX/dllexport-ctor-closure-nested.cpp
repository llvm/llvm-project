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

// Member-level dllexport on a nested default constructor needs constructor
// closure default arguments before the enclosing class is emitted.
struct MemberExportedCtorClosureOuter {
  struct MemberExportedCtorClosureInner {
    __declspec(dllexport) MemberExportedCtorClosureInner(
        const HasImplicitDtor1 &v = {}) {}
  };
};

// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FMemberExportedCtorClosureInner@MemberExportedCtorClosureOuter@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat

// Member-level dllexport on a class template specialization must build the
// constructor closure default arguments after instantiation.
template <typename T>
struct MemberExportedCtorClosureTemplate {
  __declspec(dllexport)
  MemberExportedCtorClosureTemplate(const T &v = {}) {}
};

MemberExportedCtorClosureTemplate<HasImplicitDtor1>
    MemberExportedCtorClosureTemplateInstance{{}};

// CHECK-DAG: define weak_odr dso_local dllexport x86_thiscallcc void @"??_F?$MemberExportedCtorClosureTemplate@UHasImplicitDtor1@@@@QAEXXZ"

// Explicit instantiation builds constructor closure default arguments while
// instantiating the constructor definition.
struct ExplicitInstantiationOuter {
  template <typename T>
  struct Nested {
    __declspec(dllexport)
    Nested(const HasImplicitDtor1 &v = {}) {}
  };
};

template struct ExplicitInstantiationOuter::Nested<int>;

// CHECK-DAG: define weak_odr dso_local dllexport x86_thiscallcc void @"??_F?$Nested@H@ExplicitInstantiationOuter@@QAEXXZ"
