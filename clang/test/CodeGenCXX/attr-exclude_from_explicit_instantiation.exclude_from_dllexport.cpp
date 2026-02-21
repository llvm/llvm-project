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
struct C {
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

template <class T> void C<T>::to_be_exported() {}
template <class T> void C<T>::to_be_exported_explicitly() {}
template <class T> void C<T>::not_to_be_exported() {}
template <class T> void C<T>::not_to_be_instantiated() {}

// Attach the attribute to class template declaration instead of instantiation declaration.
template <class T>
struct __declspec(dllexport) D {
  // This will be exported if and only if no explicit instantiations are provided.
  EXCLUDE_ATTR void to_be_exported_iff_no_explicit_instantiation();
};

template <class T> void D<T>::to_be_exported_iff_no_explicit_instantiation() {}

// Interaction with VTables.
template <class T>
struct E {
  // This will be instanciated by the explicit template instantiation definition.
  virtual void to_be_exported();

  // This will be instantiated by the VTable definition, regardless of
  // `exclude_from_explicit_instantiation`.
  // The dllexport attribute won't be inherited.
  EXCLUDE_ATTR virtual void to_be_instantiated();

  // This too, but will be exported by the member attribute.
  EXCLUDE_ATTR __declspec(dllexport) virtual void to_be_exported_explicitly();
};

template <class T> void E<T>::to_be_exported() {}
template <class T> void E<T>::to_be_instantiated() {}
template <class T> void E<T>::to_be_exported_explicitly() {}

// MSC: $"?to_be_exported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$E@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$E@I@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?not_to_be_exported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_iff_no_explicit_instantiation@?$D@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_iff_no_explicit_instantiation@?$D@I@@QEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$E@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$E@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$E@I@@UEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$E@I@@UEAAXXZ" = comdat any
// GNU: $_ZN1CIiE14to_be_exportedEv = comdat any
// GNU: $_ZN1EIiE14to_be_exportedEv = comdat any
// GNU: $_ZN1EIjE14to_be_exportedEv = comdat any
// GNU: $_ZN1CIiE25to_be_exported_explicitlyEv = comdat any
// GNU: $_ZN1CIiE18not_to_be_exportedEv = comdat any
// GNU: $_ZN1DIiE44to_be_exported_iff_no_explicit_instantiationEv = comdat any
// GNU: $_ZN1DIjE44to_be_exported_iff_no_explicit_instantiationEv = comdat any
// GNU: $_ZN1EIiE18to_be_instantiatedEv = comdat any
// GNU: $_ZN1EIiE25to_be_exported_explicitlyEv = comdat any
// GNU: $_ZN1EIjE18to_be_instantiatedEv = comdat any
// GNU: $_ZN1EIjE25to_be_exported_explicitlyEv = comdat any

// MSC: @0 = private unnamed_addr constant {{.*}}, comdat($"??_7?$E@H@@6B@")
// MSC: @1 = private unnamed_addr constant {{.*}}, comdat($"??_7?$E@I@@6B@")
// MSC: @"??_7?$E@H@@6B@" = dllexport unnamed_addr
// MSC: @"??_7?$E@I@@6B@" = unnamed_addr
// GNU: @_ZTV1EIiE = weak_odr dso_local dllexport unnamed_addr constant {{.*}}, comdat
// GNU: @_ZTV1EIjE = weak_odr dso_local unnamed_addr constant {{.*}}, comdat

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$C@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$C@H@@QEAAAEAU0@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1CIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1CIiEaSEOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1CIiE14to_be_exportedEv
template struct __declspec(dllexport) C<int>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$D@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$D@H@@QEAAAEAU0@$$QEAU0@@Z"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1DIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1DIiEaSEOS0_
template struct D<int>; // No dllexport here.
// Don't provide explicit instantiation for D<unsigned>.

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$E@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$E@H@@QEAAAEAU0@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$E@H@@QEAA@XZ"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$E@H@@QEAA@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??0?$E@H@@QEAA@$$QEAU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$E@H@@UEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1EIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1EIiEaSEOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiEC2Ev
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiEC1Ev
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiEC2ERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiEC1ERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiEC2EOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiEC1EOS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1EIiE14to_be_exportedEv
template struct __declspec(dllexport) E<int>;

// MSC: define weak_odr dso_local{{.*}} void @"?to_be_exported@?$E@I@@UEAAXXZ"
// GNU: define weak_odr dso_local{{.*}} void @_ZN1EIjE14to_be_exportedEv
template struct E<unsigned int>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$D@I@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$D@I@@QEAAAEAU0@$$QEAU0@@Z"

void use() {
  C<int> c;

  // MSC: call void @"?to_be_exported_explicitly@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE25to_be_exported_explicitlyEv
  c.to_be_exported_explicitly(); // implicitly instantiated here

  // MSC: call void @"?not_to_be_exported@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE18not_to_be_exportedEv
  c.not_to_be_exported(); // implicitly instantiated here

  D<int> di;

  // MSC: call void @"?to_be_exported_iff_no_explicit_instantiation@?$D@H@@QEAAXXZ"
  // GNU: call void @_ZN1DIiE44to_be_exported_iff_no_explicit_instantiationEv
  di.to_be_exported_iff_no_explicit_instantiation(); // implicitly instantiated here

  D<unsigned> dj;

  // MSC: call void @"?to_be_exported_iff_no_explicit_instantiation@?$D@I@@QEAAXXZ"
  // GNU: call void @_ZN1DIjE44to_be_exported_iff_no_explicit_instantiationEv
  dj.to_be_exported_iff_no_explicit_instantiation(); // implicitly instantiated here
}

// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported_explicitly@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1CIiE25to_be_exported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?not_to_be_exported@?$C@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1CIiE18not_to_be_exportedEv

// MSC: define linkonce_odr dso_local void @"?to_be_exported_iff_no_explicit_instantiation@?$D@H@@QEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_iff_no_explicit_instantiation@?$D@I@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1DIiE44to_be_exported_iff_no_explicit_instantiationEv
// GNU: define weak_odr dso_local dllexport void @_ZN1DIjE44to_be_exported_iff_no_explicit_instantiationEv

// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$E@H@@UEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_explicitly@?$E@H@@UEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1EIiE18to_be_instantiatedEv
// GNU: define weak_odr dso_local dllexport void @_ZN1EIiE25to_be_exported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$E@I@@UEAAXXZ"
// MSC: define weak_odr dso_local dllexport void @"?to_be_exported_explicitly@?$E@I@@UEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1EIjE18to_be_instantiatedEv
// GNU: define weak_odr dso_local dllexport void @_ZN1EIjE25to_be_exported_explicitlyEv
