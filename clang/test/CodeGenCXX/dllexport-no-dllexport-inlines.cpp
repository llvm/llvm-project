// RUN: %clang_cc1 %s -fms-extensions -triple x86_64-windows-msvc       \
// RUN:     -disable-llvm-passes                                        \
// RUN:     -fno-dllexport-inlines -emit-llvm -O1 -o - |                \
// RUN:     FileCheck --check-prefix=CHECK --check-prefix=NOEXPORTINLINE %s

// RUN: %clang_cc1 %s -fms-extensions -triple x86_64-windows-msvc       \
// RUN:     -disable-llvm-passes                                        \
// RUN:     -emit-llvm -O1 -o - |                                       \
// RUN:     FileCheck --check-prefix=CHECK  --check-prefix=EXPORTINLINE %s


struct __declspec(dllexport) ExportedClass {

  // NOEXPORTINLINE-DAG: define linkonce_odr dso_local void @"?InclassDefFunc@ExportedClass@@
  // EXPORTINLINE-DAG: define weak_odr dso_local dllexport void @"?InclassDefFunc@ExportedClass@@
  void InclassDefFunc() {}

  // CHECK-DAG: define weak_odr dso_local dllexport i32 @"?InclassDefFuncWithStaticVariable@ExportedClass@@QEAAHXZ"
  int InclassDefFuncWithStaticVariable() {
    // CHECK-DAG: @"?static_variable@?1??InclassDefFuncWithStaticVariable@ExportedClass@@QEAAHXZ@4HA" = weak_odr dso_local dllexport global i32 0, comdat, align 4
    static int static_variable = 0;
    ++static_variable;
    return static_variable;
  }

  // CHECK-DAG: define weak_odr dso_local dllexport i32 @"?InclassDefFunctWithLambdaStaticVariable@ExportedClass@@QEAAHXZ"
  int InclassDefFunctWithLambdaStaticVariable() {
    // CHECK-DAG: @"?static_x@?2???R<lambda_1>@?0??InclassDefFunctWithLambdaStaticVariable@ExportedClass@@QEAAHXZ@QEBA?A?<auto>@@XZ@4HA" = weak_odr dso_local dllexport global i32 0, comdat, align 4
    return ([]() { static int static_x; return ++static_x; })();
  }

  // NOEXPORTINLINE-DAG: define linkonce_odr dso_local void @"?InlineOutclassDefFunc@ExportedClass@@QEAAXXZ
  // EXPORTINLINE-DAG: define weak_odr dso_local dllexport void @"?InlineOutclassDefFunc@ExportedClass@@QEAAXXZ
  inline void InlineOutclassDefFunc();

  // CHECK-DAG: define weak_odr dso_local dllexport i32 @"?InlineOutclassDefFuncWithStaticVariable@ExportedClass@@QEAAHXZ"
  inline int InlineOutclassDefFuncWithStaticVariable();

  // CHECK-DAG: define dso_local dllexport void @"?OutoflineDefFunc@ExportedClass@@QEAAXXZ"
  void OutoflineDefFunc();
};

void ExportedClass::OutoflineDefFunc() {}

inline void ExportedClass::InlineOutclassDefFunc() {}

inline int ExportedClass::InlineOutclassDefFuncWithStaticVariable() {
  static int static_variable = 0;
  return ++static_variable;
}

void ExportedClassUser() {
  ExportedClass a;
  a.InclassDefFunc();
  a.InlineOutclassDefFunc();
}

template<typename T>
struct __declspec(dllexport) TemplateExportedClass {
  void InclassDefFunc() {}

  int InclassDefFuncWithStaticVariable() {
    static int static_x = 0;
    return ++static_x;
  }
};

class A11{};
class B22{};

// CHECK-DAG: define weak_odr dso_local dllexport void @"?InclassDefFunc@?$TemplateExportedClass@VA11@@@@QEAAXXZ"
// CHECK-DAG: define weak_odr dso_local dllexport i32 @"?InclassDefFuncWithStaticVariable@?$TemplateExportedClass@VA11@@@@QEAAHXZ"
// CHECK-DAG: @"?static_x@?2??InclassDefFuncWithStaticVariable@?$TemplateExportedClass@VA11@@@@QEAAHXZ@4HA" = weak_odr dso_local dllexport global i32 0, comdat, align 4
template class TemplateExportedClass<A11>;

// NOEXPORTINLINE-DAG: define linkonce_odr dso_local void @"?InclassDefFunc@?$TemplateExportedClass@VB22@@@@QEAAXXZ"
// EXPORTINLINE-DAG: define weak_odr dso_local dllexport void @"?InclassDefFunc@?$TemplateExportedClass@VB22@@@@QEAAXXZ
// CHECK-DAG: define weak_odr dso_local dllexport i32 @"?InclassDefFuncWithStaticVariable@?$TemplateExportedClass@VB22@@@@QEAAHXZ"
// CHECK-DAG: @"?static_x@?2??InclassDefFuncWithStaticVariable@?$TemplateExportedClass@VB22@@@@QEAAHXZ@4HA" = weak_odr dso_local dllexport global i32 0, comdat, align 4
TemplateExportedClass<B22> b22;

void TemplateExportedClassUser() {
  b22.InclassDefFunc();
  b22.InclassDefFuncWithStaticVariable();
}


template<typename T>
struct TemplateNoAttributeClass {
  void InclassDefFunc() {}
  int InclassDefFuncWithStaticLocal() {
    static int static_x;
    return ++static_x;
  }
};

// CHECK-DAG: define weak_odr dso_local dllexport void @"?InclassDefFunc@?$TemplateNoAttributeClass@VA11@@@@QEAAXXZ"
// CHECK-DAG: define weak_odr dso_local dllexport i32 @"?InclassDefFuncWithStaticLocal@?$TemplateNoAttributeClass@VA11
// CHECK-DAG: @"?static_x@?2??InclassDefFuncWithStaticLocal@?$TemplateNoAttributeClass@VA11@@@@QEAAHXZ@4HA" = weak_odr dso_local dllexport global i32 0, comdat, align 4
template class __declspec(dllexport) TemplateNoAttributeClass<A11>;

// CHECK-DAG: define available_externally dllimport void @"?InclassDefFunc@?$TemplateNoAttributeClass@VB22@@@@QEAAXXZ"
// CHECK-DAG: define available_externally dllimport i32 @"?InclassDefFuncWithStaticLocal@?$TemplateNoAttributeClass@VB22@@@@QEAAHXZ"
// CHECK-DAG: @"?static_x@?2??InclassDefFuncWithStaticLocal@?$TemplateNoAttributeClass@VB22@@@@QEAAHXZ@4HA" = available_externally dllimport global i32 0, align 4
extern template class __declspec(dllimport) TemplateNoAttributeClass<B22>;

void TemplateNoAttributeClassUser() {
  TemplateNoAttributeClass<B22> b22;
  b22.InclassDefFunc();
  b22.InclassDefFuncWithStaticLocal();
}

struct __declspec(dllimport) ImportedClass {
  // NOEXPORTINLINE-DAG: define linkonce_odr dso_local void @"?InClassDefFunc@ImportedClass@@QEAAXXZ"
  // EXPORTINLINE-DAG: define available_externally dllimport void @"?InClassDefFunc@ImportedClass@@QEAAXXZ"
  void InClassDefFunc() {}

  // EXPORTINLINE-DAG: define available_externally dllimport i32 @"?InClassDefFuncWithStaticVariable@ImportedClass@@QEAAHXZ"
  // NOEXPORTINLINE-DAG: define linkonce_odr dso_local i32 @"?InClassDefFuncWithStaticVariable@ImportedClass@@QEAAHXZ"
  int InClassDefFuncWithStaticVariable() {
    // CHECK-DAG: @"?static_variable@?1??InClassDefFuncWithStaticVariable@ImportedClass@@QEAAHXZ@4HA" = available_externally dllimport global i32 0, align 4
    static int static_variable = 0;
    ++static_variable;
    return static_variable;
  }
};

int InClassDefFuncUser() {
  // This is necessary for declare statement of ImportedClass::InClassDefFunc().
  ImportedClass c;
  c.InClassDefFunc();
  return c.InClassDefFuncWithStaticVariable();
}
