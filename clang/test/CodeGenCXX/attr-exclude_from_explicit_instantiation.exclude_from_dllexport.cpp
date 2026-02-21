// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=MSC --implicit-check-not=to_be_ --implicit-check-not=dllexport
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllexport
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllexport

// Test that __declspec(dllexport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute unless marked as dllexport explicitly.

// Silence --implicit-check-not=dllexport.
// MSC: ModuleID = {{.*}}exclude_from_dllexport.cpp
// MSC: source_filename = {{.*}}exclude_from_dllexport.cpp
// GNU: ModuleID = {{.*}}exclude_from_dllexport.cpp
// GNU: source_filename = {{.*}}exclude_from_dllexport.cpp

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct BasicCase {
  // This will be instantiated explicitly as an exported function because it
  // inherits dllexport from the class instantiation.
  void to_be_exported();

  // This will be instantiated implicitly as an exported function because it is
  // marked as dllexport explicitly.
  EXCLUDE_ATTR __declspec(dllexport) void to_be_exported_explicitly();

  // This will be instantiated implicitly but won't be exported.
  EXCLUDE_ATTR void not_to_be_exported();

  // This won't be instantiated.
  EXCLUDE_ATTR void not_to_be_instantiated();
};

// Member functions can't be inlined since clang in MinGW mode doesn't export/import them that are inlined.
template <class T> void BasicCase<T>::to_be_exported() {}
template <class T> void BasicCase<T>::to_be_exported_explicitly() {}
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
  EXCLUDE_ATTR __declspec(dllexport) virtual void to_be_exported_explicitly();
};

template <class T> void Polymorphic<T>::to_be_exported() {}
template <class T> void Polymorphic<T>::to_be_instantiated() {}
template <class T> void Polymorphic<T>::to_be_exported_explicitly() {}

// MSC: $"?to_be_exported@?$BasicCase@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$Polymorphic@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$Polymorphic@I@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$BasicCase@H@@QEAAXXZ" = comdat any
// MSC: $"?not_to_be_exported@?$BasicCase@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@I@@QEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$Polymorphic@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$Polymorphic@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$Polymorphic@I@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$Polymorphic@I@@UEAAXXZ" = comdat any
// GNU: $_ZN9BasicCaseIiE14to_be_exportedEv = comdat any
// GNU: $_ZN11PolymorphicIiE14to_be_exportedEv = comdat any
// GNU: $_ZN11PolymorphicIjE14to_be_exportedEv = comdat any
// GNU: $_ZN9BasicCaseIiE25to_be_exported_explicitlyEv = comdat any
// GNU: $_ZN9BasicCaseIiE18not_to_be_exportedEv = comdat any
// GNU: $_ZN19ExportWholeTemplateIiE44to_be_exported_iff_no_explicit_instantiationEv = comdat any
// GNU: $_ZN19ExportWholeTemplateIjE44to_be_exported_iff_no_explicit_instantiationEv = comdat any
// GNU: $_ZN11PolymorphicIiE18to_be_instantiatedEv = comdat any
// GNU: $_ZN11PolymorphicIiE25to_be_exported_explicitlyEv = comdat any
// GNU: $_ZN11PolymorphicIjE18to_be_instantiatedEv = comdat any
// GNU: $_ZN11PolymorphicIjE25to_be_exported_explicitlyEv = comdat any

// MSC: @0 = private unnamed_addr constant {{.*}}, comdat($"??_7?$Polymorphic@H@@6B@")
// MSC: @1 = private unnamed_addr constant {{.*}}, comdat($"??_7?$Polymorphic@I@@6B@")
// MSC: @"??_7?$Polymorphic@H@@6B@" = dllexport unnamed_addr
// MSC: @"??_7?$Polymorphic@I@@6B@" = unnamed_addr
// GNU: @_ZTV11PolymorphicIiE = weak_odr dso_local dllexport unnamed_addr constant {{.*}}, comdat
// GNU: @_ZTV11PolymorphicIjE = weak_odr dso_local unnamed_addr constant {{.*}}, comdat

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$BasicCase@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$BasicCase@H@@QEAAAEAU0@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$BasicCase@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN9BasicCaseIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN9BasicCaseIiEaSEOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN9BasicCaseIiE14to_be_exportedEv
template struct __declspec(dllexport) BasicCase<int>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@H@@QEAAAEAU0@$$QEAU0@@Z"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN19ExportWholeTemplateIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN19ExportWholeTemplateIiEaSEOS0_
template struct ExportWholeTemplate<int>; // No dllexport here.
// Don't provide explicit instantiation for ExportWholeTemplate<unsigned>.

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$Polymorphic@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$Polymorphic@H@@QEAAAEAU0@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$Polymorphic@H@@QEAA@XZ"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$Polymorphic@H@@QEAA@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$Polymorphic@H@@QEAA@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$Polymorphic@H@@UEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN11PolymorphicIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN11PolymorphicIiEaSEOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiEC2Ev
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiEC1Ev
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiEC2ERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiEC1ERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiEC2EOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiEC1EOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN11PolymorphicIiE14to_be_exportedEv
template struct __declspec(dllexport) Polymorphic<int>;

// MSC: define weak_odr dso_local{{.*}} void @"?to_be_exported@?$Polymorphic@I@@UEAAXXZ"
// GNU: define weak_odr dso_local{{.*}} void @_ZN11PolymorphicIjE14to_be_exportedEv
template struct Polymorphic<unsigned int>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@I@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$ExportWholeTemplate@I@@QEAAAEAU0@$$QEAU0@@Z"

void use() {
  BasicCase<int> c;

  // MSC: call void @"?to_be_exported_explicitly@?$BasicCase@H@@QEAAXXZ"
  // GNU: call void @_ZN9BasicCaseIiE25to_be_exported_explicitlyEv
  c.to_be_exported_explicitly(); // implicitly instantiated here

  // MSC: call void @"?not_to_be_exported@?$BasicCase@H@@QEAAXXZ"
  // GNU: call void @_ZN9BasicCaseIiE18not_to_be_exportedEv
  c.not_to_be_exported(); // implicitly instantiated here

  ExportWholeTemplate<int> di;

  // MSC: call void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@H@@QEAAXXZ"
  // GNU: call void @_ZN19ExportWholeTemplateIiE44to_be_exported_iff_no_explicit_instantiationEv
  di.to_be_exported_iff_no_explicit_instantiation(); // implicitly instantiated here

  ExportWholeTemplate<unsigned> dj;

  // MSC: call void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@I@@QEAAXXZ"
  // GNU: call void @_ZN19ExportWholeTemplateIjE44to_be_exported_iff_no_explicit_instantiationEv
  dj.to_be_exported_iff_no_explicit_instantiation(); // implicitly instantiated here
}

// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported_explicitly@?$BasicCase@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN9BasicCaseIiE25to_be_exported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?not_to_be_exported@?$BasicCase@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN9BasicCaseIiE18not_to_be_exportedEv

// MSC: define linkonce_odr dso_local void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@H@@QEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_iff_no_explicit_instantiation@?$ExportWholeTemplate@I@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN19ExportWholeTemplateIiE44to_be_exported_iff_no_explicit_instantiationEv
// GNU: define weak_odr dso_local dllexport void @_ZN19ExportWholeTemplateIjE44to_be_exported_iff_no_explicit_instantiationEv

// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$Polymorphic@H@@UEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_explicitly@?$Polymorphic@H@@UEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN11PolymorphicIiE18to_be_instantiatedEv
// GNU: define weak_odr dso_local dllexport void @_ZN11PolymorphicIiE25to_be_exported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$Polymorphic@I@@UEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_explicitly@?$Polymorphic@I@@UEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN11PolymorphicIjE18to_be_instantiatedEv
// GNU: define weak_odr dso_local dllexport void @_ZN11PolymorphicIjE25to_be_exported_explicitlyEv
