// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=MSC
// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --implicit-check-not=notToBeInstantiated
//
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --implicit-check-not=notToBeInstantiated
//
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --implicit-check-not=notToBeInstantiated

// Because --implicit-check-not doesn't work with -DAG checks, negative checks
// are performed on another independent path.

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

struct NoAttrTag {};
struct WithExportTag {};
struct ImplicitTag {};

// Test that __declspec(dllexport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute.
template <class T>
struct BasicCase {
  void noAttrMethod() {}
  EXCLUDE_ATTR void excludedMethod() {}
  EXCLUDE_ATTR __declspec(dllexport) void excludedExportedMethod() {}
  EXCLUDE_ATTR void notToBeInstantiated() {}
  EXCLUDE_ATTR __declspec(dllexport) void notToBeInstantiated_withExport() {}
};

/// Test that an exported explicit instantiation definition causes to export
/// non-exclued methods (i.e., noAttrMethod) only.
template struct __declspec(dllexport) BasicCase<WithExportTag>;
// MSC-DAG: define weak_odr dso_local dllexport void @"?noAttrMethod@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
// GNU-DAG: define weak_odr dso_local dllexport void @_ZN9BasicCaseI13WithExportTagE12noAttrMethodEv

/// Test that a non-exported explicit instantiation definition instantiates
/// non-exclued methods but not exports.
template struct BasicCase<NoAttrTag>;
// MSC-DAG: define weak_odr dso_local void @"?noAttrMethod@?$BasicCase@UNoAttrTag@@@@QEAAXXZ"
// GNU-DAG: define weak_odr dso_local void @_ZN9BasicCaseI9NoAttrTagE12noAttrMethodEv

/// Test that an excluded method isn't exported even if the previous explicit
/// instantiation definition or the method itself is exported.
/// A never-called method `notToBeInstantiated` makes sure that an excluded
/// method isn't instantiated unexpectedly.
void useBasicCase() {
  BasicCase<WithExportTag>().excludedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedMethod@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN9BasicCaseI13WithExportTagE14excludedMethodEv

  BasicCase<WithExportTag>().excludedExportedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedExportedMethod@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN9BasicCaseI13WithExportTagE22excludedExportedMethodEv

  BasicCase<NoAttrTag>().excludedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedMethod@?$BasicCase@UNoAttrTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN9BasicCaseI9NoAttrTagE14excludedMethodEv

  BasicCase<NoAttrTag>().excludedExportedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedExportedMethod@?$BasicCase@UNoAttrTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN9BasicCaseI9NoAttrTagE22excludedExportedMethodEv

  BasicCase<ImplicitTag>().excludedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedMethod@?$BasicCase@UImplicitTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN9BasicCaseI11ImplicitTagE14excludedMethodEv

  BasicCase<ImplicitTag>().excludedExportedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedExportedMethod@?$BasicCase@UImplicitTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN9BasicCaseI11ImplicitTagE22excludedExportedMethodEv
}

// Test that a class-level dllexport attribute won't affect to excluded methods.
template <class T>
struct __declspec(dllexport) ExportWholeTemplate {
  void noAttrMethod() {}
  EXCLUDE_ATTR void excludedMethod() {}
  EXCLUDE_ATTR void excludedNoinlineMethod();
  EXCLUDE_ATTR void notToBeInstantiated() {}
};

// MSVC and MinGW disagree on whether an inline method of a class-level exported
// template should be exported.
template <typename T> void ExportWholeTemplate<T>::excludedNoinlineMethod() {}

template struct ExportWholeTemplate<NoAttrTag>;
// MSC-DAG: define weak_odr dso_local dllexport void @"?noAttrMethod@?$ExportWholeTemplate@UNoAttrTag@@@@QEAAXXZ"
// GNU-DAG: define weak_odr dso_local dllexport void @_ZN19ExportWholeTemplateI9NoAttrTagE12noAttrMethodEv

void useExportWholeTemplate() {
  ExportWholeTemplate<NoAttrTag>().excludedMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedMethod@?$ExportWholeTemplate@UNoAttrTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN19ExportWholeTemplateI9NoAttrTagE14excludedMethodEv

  ExportWholeTemplate<NoAttrTag>().excludedNoinlineMethod();
  // MSC-DAG: define linkonce_odr dso_local void @"?excludedNoinlineMethod@?$ExportWholeTemplate@UNoAttrTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN19ExportWholeTemplateI9NoAttrTagE22excludedNoinlineMethodEv

  ExportWholeTemplate<ImplicitTag>().excludedMethod();
  // MSC-DAG: define weak_odr dso_local dllexport void @"?excludedMethod@?$ExportWholeTemplate@UImplicitTag@@@@QEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN19ExportWholeTemplateI11ImplicitTagE14excludedMethodEv

  ExportWholeTemplate<ImplicitTag>().excludedNoinlineMethod();
  // MSC-DAG: define weak_odr dso_local dllexport void @"?excludedNoinlineMethod@?$ExportWholeTemplate@UImplicitTag@@@@QEAAXXZ"
  // GNU-DAG: define weak_odr dso_local dllexport void @_ZN19ExportWholeTemplateI11ImplicitTagE22excludedNoinlineMethodEv
}

// Interaction with VTables.
template <class T>
struct Polymorphic {
  EXCLUDE_ATTR explicit Polymorphic() = default;
  virtual void noAttrVirtualMethod() {}
  EXCLUDE_ATTR virtual void excludedVirtualMethod() {}
  EXCLUDE_ATTR __declspec(dllexport) virtual void excludedExportedVirtualMethod() {}
  EXCLUDE_ATTR void notToBeInstantiated() {}
  EXCLUDE_ATTR __declspec(dllexport) void notToBeInstantiated_withExport() {}
};

template struct __declspec(dllexport) Polymorphic<WithExportTag>;
// MSC-DAG: @"??_7?$Polymorphic@UWithExportTag@@@@6B@" = dllexport unnamed_addr
// GNU-DAG: @_ZTV11PolymorphicI13WithExportTagE = weak_odr dso_local dllexport unnamed_addr

// MSC-DAG: define weak_odr dso_local dllexport void @"?noAttrVirtualMethod@?$Polymorphic@UWithExportTag@@@@UEAAXXZ"
// GNU-DAG: define weak_odr dso_local dllexport void @_ZN11PolymorphicI13WithExportTagE19noAttrVirtualMethodEv

// MSC-DAG: define linkonce_odr dso_local void @"?excludedVirtualMethod@?$Polymorphic@UWithExportTag@@@@UEAAXXZ"
// GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI13WithExportTagE21excludedVirtualMethodEv

// MSC-DAG: define linkonce_odr dso_local void @"?excludedExportedVirtualMethod@?$Polymorphic@UWithExportTag@@@@UEAAXXZ"
// GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI13WithExportTagE29excludedExportedVirtualMethodEv

template struct Polymorphic<NoAttrTag>;
// MSC-DAG: @"??_7?$Polymorphic@UNoAttrTag@@@@6B@" = unnamed_addr
// GNU-DAG: @_ZTV11PolymorphicI9NoAttrTagE = weak_odr dso_local unnamed_addr

// MSC-DAG: define weak_odr dso_local void @"?noAttrVirtualMethod@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ"
// GNU-DAG: define weak_odr dso_local void @_ZN11PolymorphicI9NoAttrTagE19noAttrVirtualMethodEv

// MSC-DAG: define linkonce_odr dso_local void @"?excludedVirtualMethod@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ"
// GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI9NoAttrTagE21excludedVirtualMethodEv

// MSC-DAG: define linkonce_odr dso_local void @"?excludedExportedVirtualMethod@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ"
// GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI9NoAttrTagE29excludedExportedVirtualMethodEv

void usePolymorphic() {
  new Polymorphic<ImplicitTag>();
  // MSC-DAG: @"??_7?$Polymorphic@UImplicitTag@@@@6B@" = unnamed_addr
  // GNU-DAG: @_ZTV11PolymorphicI11ImplicitTagE = linkonce_odr dso_local unnamed_addr

  // MSC-DAG: define linkonce_odr dso_local void @"?noAttrVirtualMethod@?$Polymorphic@UImplicitTag@@@@UEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI11ImplicitTagE19noAttrVirtualMethodEv

  // MSC-DAG: define linkonce_odr dso_local void @"?excludedVirtualMethod@?$Polymorphic@UImplicitTag@@@@UEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI11ImplicitTagE21excludedVirtualMethodEv

  // MSC-DAG: define linkonce_odr dso_local void @"?excludedExportedVirtualMethod@?$Polymorphic@UImplicitTag@@@@UEAAXXZ"
  // GNU-DAG: define linkonce_odr dso_local void @_ZN11PolymorphicI11ImplicitTagE29excludedExportedVirtualMethodEv
}

/// Test that the DLL attrribute wins over the exclude attribute on a
/// non-template context.
struct NonTemplateClass {
  EXCLUDE_ATTR __declspec(dllexport) void excludedExportedMethod();
};

void NonTemplateClass::excludedExportedMethod() {}
// MSC-DAG: define dso_local dllexport void @"?excludedExportedMethod@NonTemplateClass@@QEAAXXZ"
// GNU-DAG: define dso_local dllexport void @_ZN16NonTemplateClass22excludedExportedMethodEv

/// The same, but exporting whole class.
struct __declspec(dllexport) NonTemplateExportedClass {
  EXCLUDE_ATTR void excludedMethod();
};

void NonTemplateExportedClass::excludedMethod() {}
// MSC-DAG: define dso_local dllexport void @"?excludedMethod@NonTemplateExportedClass@@QEAAXXZ"
// GNU-DAG: define dso_local dllexport void @_ZN24NonTemplateExportedClass14excludedMethodEv
