// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=MSC --implicit-check-not=to_be_ --implicit-check-not=" dllexport"
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=" dllexport"
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=" dllexport"

// Test that __declspec(dllexport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute unless marked as dllexport explicitly.

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

struct NoAttrTag {};
struct WithExportTag {};
struct ImplicitTag {};

template <class T>
struct BasicCase {
  // This will be instantiated explicitly as an exported function because it
  // inherits dllexport from the class instantiation.
  void to_be_exported();

  // This will be instantiated implicitly as an exported function because it is
  // marked as dllexport explicitly.
  EXCLUDE_ATTR __declspec(dllexport) void to_be_memberwise_exported();

  // This will be instantiated implicitly but won't be exported.
  EXCLUDE_ATTR void not_to_be_exported();

  // This won't be instantiated.
  EXCLUDE_ATTR void not_to_be_instantiated();
};

// Member functions can't be inlined since clang in MinGW mode doesn't export/import them that are inlined.
template <class T> void BasicCase<T>::to_be_exported() {}
template <class T> void BasicCase<T>::to_be_memberwise_exported() {}
template <class T> void BasicCase<T>::not_to_be_exported() {}
template <class T> void BasicCase<T>::not_to_be_instantiated() {}

// Attach the attribute to class template declaration instead of instantiation declaration.
template <class T>
struct __declspec(dllexport) ExportWholeTemplate {
  // This will be exported if and only if no explicit instantiations are provided.
  EXCLUDE_ATTR void to_be_exported_iff_no_explicit_instantiation();
};

template <class T> void ExportWholeTemplate<T>::to_be_exported_iff_no_explicit_instantiation() {}

// Interaction with VTables.
template <class T>
struct Polymorphic {
  // This will be instanciated by the explicit template instantiation definition.
  virtual void to_be_exported();

  // This will be instantiated by the VTable definition, regardless of
  // `exclude_from_explicit_instantiation`.
  // The dllexport attribute won't be inherited.
  EXCLUDE_ATTR virtual void to_be_instantiated();

  // This too, but will be exported by the member attribute.
  EXCLUDE_ATTR __declspec(dllexport) virtual void to_be_memberwise_exported();
};

template <class T> void Polymorphic<T>::to_be_exported() {}
template <class T> void Polymorphic<T>::to_be_instantiated() {}
template <class T> void Polymorphic<T>::to_be_memberwise_exported() {}

// MSC: $"?to_be_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$Polymorphic@UWithExportTag@@@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ" = comdat any
// MSC: $"?to_be_memberwise_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ" = comdat any
// MSC: $"?not_to_be_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@UNoAttrTag@@@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@UImplicitTag@@@@QEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$Polymorphic@UWithExportTag@@@@UEAAXXZ" = comdat any
// MSC: $"?to_be_memberwise_exported@?$Polymorphic@UWithExportTag@@@@UEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ" = comdat any
// MSC: $"?to_be_memberwise_exported@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ" = comdat any
// GNU: $_ZN9BasicCaseI13WithExportTagE14to_be_exportedEv = comdat any
// GNU: $_ZN11PolymorphicI13WithExportTagE14to_be_exportedEv = comdat any
// GNU: $_ZN11PolymorphicI9NoAttrTagE14to_be_exportedEv = comdat any
// GNU: $_ZN9BasicCaseI13WithExportTagE25to_be_memberwise_exportedEv = comdat any
// GNU: $_ZN9BasicCaseI13WithExportTagE18not_to_be_exportedEv = comdat any
// GNU: $_ZN19ExportWholeTemplateI9NoAttrTagE44to_be_exported_iff_no_explicit_instantiationEv = comdat any
// GNU: $_ZN19ExportWholeTemplateI11ImplicitTagE44to_be_exported_iff_no_explicit_instantiationEv = comdat any
// GNU: $_ZN11PolymorphicI13WithExportTagE18to_be_instantiatedEv = comdat any
// GNU: $_ZN11PolymorphicI13WithExportTagE25to_be_memberwise_exportedEv = comdat any
// GNU: $_ZN11PolymorphicI9NoAttrTagE18to_be_instantiatedEv = comdat any
// GNU: $_ZN11PolymorphicI9NoAttrTagE25to_be_memberwise_exportedEv = comdat any

// MSC: @0 = private unnamed_addr constant {{.*}}, comdat($"??_7?$Polymorphic@UWithExportTag@@@@6B@")
// MSC: @1 = private unnamed_addr constant {{.*}}, comdat($"??_7?$Polymorphic@UNoAttrTag@@@@6B@")
// MSC: @"??_7?$Polymorphic@UWithExportTag@@@@6B@" = dllexport unnamed_addr
// MSC: @"??_7?$Polymorphic@UNoAttrTag@@@@6B@" = unnamed_addr
// GNU: @_ZTV11PolymorphicI13WithExportTagE = weak_odr dso_local dllexport unnamed_addr constant {{.*}}, comdat
// GNU: @_ZTV11PolymorphicI9NoAttrTagE = weak_odr dso_local unnamed_addr constant {{.*}}, comdat

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$BasicCase@UWithExportTag@@@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$BasicCase@UWithExportTag@@@@QEAAAEAU0@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN9BasicCaseI13WithExportTagEaSERKS1_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN9BasicCaseI13WithExportTagEaSEOS1_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN9BasicCaseI13WithExportTagE14to_be_exportedEv
template struct __declspec(dllexport) BasicCase<WithExportTag>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@UNoAttrTag@@@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@UNoAttrTag@@@@QEAAAEAU0@$$QEAU0@@Z"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN19ExportWholeTemplateI9NoAttrTagEaSERKS1_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN19ExportWholeTemplateI9NoAttrTagEaSEOS1_
template struct ExportWholeTemplate<NoAttrTag>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$Polymorphic@UWithExportTag@@@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$Polymorphic@UWithExportTag@@@@QEAAAEAU0@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$Polymorphic@UWithExportTag@@@@QEAA@XZ"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$Polymorphic@UWithExportTag@@@@QEAA@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$Polymorphic@UWithExportTag@@@@QEAA@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$Polymorphic@UWithExportTag@@@@UEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN11PolymorphicI13WithExportTagEaSERKS1_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN11PolymorphicI13WithExportTagEaSEOS1_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagEC2Ev
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagEC1Ev
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagEC2ERKS1_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagEC1ERKS1_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagEC2EOS1_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagEC1EOS1_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicI13WithExportTagE14to_be_exportedEv
template struct __declspec(dllexport) Polymorphic<WithExportTag>;

// MSC: define weak_odr dso_local{{.*}} void @"?to_be_exported@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ"
// GNU: define weak_odr dso_local{{.*}} void @_ZN11PolymorphicI9NoAttrTagE14to_be_exportedEv
template struct Polymorphic<NoAttrTag>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@UImplicitTag@@@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@UImplicitTag@@@@QEAAAEAU0@$$QEAU0@@Z"

void use() {
  BasicCase<WithExportTag> c;

  // MSC: call void @"?to_be_memberwise_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
  // GNU: call void @_ZN9BasicCaseI13WithExportTagE25to_be_memberwise_exportedEv
  c.to_be_memberwise_exported(); // implicitly instantiated here

  // MSC: call void @"?not_to_be_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
  // GNU: call void @_ZN9BasicCaseI13WithExportTagE18not_to_be_exportedEv
  c.not_to_be_exported(); // implicitly instantiated here

  ExportWholeTemplate<NoAttrTag> di;

  // MSC: call void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@UNoAttrTag@@@@QEAAXXZ"
  // GNU: call void @_ZN19ExportWholeTemplateI9NoAttrTagE44to_be_exported_iff_no_explicit_instantiationEv
  di.to_be_exported_iff_no_explicit_instantiation(); // implicitly instantiated here

  ExportWholeTemplate<ImplicitTag> dj;

  // MSC: call void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@UImplicitTag@@@@QEAAXXZ"
  // GNU: call void @_ZN19ExportWholeTemplateI11ImplicitTagE44to_be_exported_iff_no_explicit_instantiationEv
  dj.to_be_exported_iff_no_explicit_instantiation(); // implicitly instantiated here
}

// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_memberwise_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN9BasicCaseI13WithExportTagE25to_be_memberwise_exportedEv

// MSC: define linkonce_odr dso_local void @"?not_to_be_exported@?$BasicCase@UWithExportTag@@@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN9BasicCaseI13WithExportTagE18not_to_be_exportedEv

// MSC: define linkonce_odr dso_local void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@UNoAttrTag@@@@QEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@UImplicitTag@@@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN19ExportWholeTemplateI9NoAttrTagE44to_be_exported_iff_no_explicit_instantiationEv
// GNU: define weak_odr dso_local dllexport void @_ZN19ExportWholeTemplateI11ImplicitTagE44to_be_exported_iff_no_explicit_instantiationEv

// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$Polymorphic@UWithExportTag@@@@UEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_memberwise_exported@?$Polymorphic@UWithExportTag@@@@UEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN11PolymorphicI13WithExportTagE18to_be_instantiatedEv
// GNU: define weak_odr dso_local dllexport void @_ZN11PolymorphicI13WithExportTagE25to_be_memberwise_exportedEv

// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_memberwise_exported@?$Polymorphic@UNoAttrTag@@@@UEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN11PolymorphicI9NoAttrTagE18to_be_instantiatedEv
// GNU: define weak_odr dso_local dllexport void @_ZN11PolymorphicI9NoAttrTagE25to_be_memberwise_exportedEv
