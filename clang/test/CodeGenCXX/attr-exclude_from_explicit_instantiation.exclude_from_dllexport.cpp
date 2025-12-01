// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=MSC --implicit-check-not=to_be_ --implicit-check-not=dllexport
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllexport
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllexport

// Test that __declspec(dllexport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute unless marked as dllexport explicitly.

// MSC: ModuleID = {{.*}}exclude_from_dllexport.cpp
// MSC: source_filename = {{.*}}exclude_from_dllexport.cpp
// GNU: ModuleID = {{.*}}exclude_from_dllexport.cpp
// GNU: source_filename = {{.*}}exclude_from_dllexport.cpp

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct C {
  // This will be instantiated explicitly as an exported function because it
  // inherits dllexport from the class instantiation.
  void to_be_exported() noexcept;

  // This will be instantiated implicitly as an exported function because it is
  // marked as dllexport explicitly.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION __declspec(dllexport) void to_be_exported_explicitly() noexcept;

  // This will be instantiated implicitly but won't be exported.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void not_to_be_exported() noexcept;

  // This won't be instantiated.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void not_to_be_instantiated() noexcept;
};

template <class T> void C<T>::to_be_exported() noexcept {}
template <class T> void C<T>::to_be_exported_explicitly() noexcept {}
template <class T> void C<T>::not_to_be_exported() noexcept {}
template <class T> void C<T>::not_to_be_instantiated() noexcept {}

// Attach the attribute to class template declaration instead of instantiation declaration.
template <class T>
struct __declspec(dllexport) D {
  // This should be exported by the class-level attribute.
  void to_be_exported() noexcept;

  // This also should be exported by the class-level attribute but currently not.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void also_to_be_exported() noexcept;
};

template <class T> void D<T>::to_be_exported() noexcept {}
template <class T> void D<T>::also_to_be_exported() noexcept {}

// MSC: $"?to_be_exported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported@?$D@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?not_to_be_exported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?also_to_be_exported@?$D@H@@QEAAXXZ" = comdat any
// GNU: $_ZN1CIiE14to_be_exportedEv = comdat any
// GNU: $_ZN1DIiE14to_be_exportedEv = comdat any
// GNU: $_ZN1CIiE25to_be_exported_explicitlyEv = comdat any
// GNU: $_ZN1CIiE18not_to_be_exportedEv = comdat any
// GNU: $_ZN1DIiE19also_to_be_exportedEv = comdat any

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$C@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1CIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1CIiE14to_be_exportedEv
template struct __declspec(dllexport) C<int>;

// MSC: define weak_odr dso_local dllexport{{.*}} ptr @"??4?$D@H@@QEAAAEAU0@AEBU0@@Z"
// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$D@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} ptr @_ZN1DIiEaSERKS0_
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1DIiE14to_be_exportedEv
template struct D<int>;

void use() {
  C<int> c;

  // MSC: call void @"?to_be_exported_explicitly@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE25to_be_exported_explicitlyEv
  c.to_be_exported_explicitly(); // implicitly instantiated here

  // MSC: call void @"?not_to_be_exported@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE18not_to_be_exportedEv
  c.not_to_be_exported(); // implicitly instantiated here

  D<int> d;

  // MSC: call void @"?also_to_be_exported@?$D@H@@QEAAXXZ"
  // GNU: call void @_ZN1DIiE19also_to_be_exportedEv
  d.also_to_be_exported(); // implicitly instantiated here
}

// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported_explicitly@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1CIiE25to_be_exported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?not_to_be_exported@?$C@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1CIiE18not_to_be_exportedEv

// MSC: define linkonce_odr dso_local void @"?also_to_be_exported@?$D@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1DIiE19also_to_be_exportedEv
