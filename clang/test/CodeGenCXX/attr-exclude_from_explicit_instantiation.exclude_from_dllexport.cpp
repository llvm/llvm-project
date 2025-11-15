// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | FileCheck %s --check-prefixes=MSC --implicit-check-not=to_be_
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_

// Test that __declspec(dllexport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute unless marked as dllexport explicitly.

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct C {
  // This will be instantiated explicitly as an exported function because it
  // inherits dllexport from the class instantiation.
  void to_be_exported() noexcept;

  // This will be instantiated implicitly as an exported function because it is
  // marked as dllexport explicitly.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION __declspec(dllexport) void to_be_exported_explicitly() noexcept;

  // This will be instantiated implicitly as an exported function unintentionally.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void not_to_be_exported() noexcept;

  // This won't be instantiated.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void not_to_be_instantiated() noexcept;
};

template <class T> void C<T>::to_be_exported() noexcept {}
template <class T> void C<T>::to_be_exported_explicitly() noexcept {}
template <class T> void C<T>::not_to_be_exported() noexcept {}
template <class T> void C<T>::not_to_be_instantiated() noexcept {}

// MSC: $"?to_be_exported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_exported_explicitly@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?not_to_be_exported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?not_to_be_instantiated@?$C@H@@QEAAXXZ" = comdat any
// GNU: $_ZN1CIiE14to_be_exportedEv = comdat any
// GNU: $_ZN1CIiE25to_be_exported_explicitlyEv = comdat any
// GNU: $_ZN1CIiE18not_to_be_exportedEv = comdat any
// GNU: $_ZN1CIiE22not_to_be_instantiatedEv = comdat any

// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1CIiE14to_be_exportedEv
template struct __declspec(dllexport) C<int>;

void use() {
  C<int> c;

  // MSC: call void @"?to_be_exported_explicitly@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE25to_be_exported_explicitlyEv
  c.to_be_exported_explicitly(); // implicitly instantiated here

  // MSC: call void @"?not_to_be_exported@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE18not_to_be_exportedEv
  c.not_to_be_exported(); // implicitly instantiated here
};

// MSC: define weak_odr dso_local dllexport{{.*}} void @"?to_be_exported_explicitly@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport{{.*}} void @_ZN1CIiE25to_be_exported_explicitlyEv

// MSC: define weak_odr dso_local dllexport void @"?not_to_be_exported@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport void @_ZN1CIiE18not_to_be_exportedEv

// MSC: define weak_odr dso_local dllexport void @"?not_to_be_instantiated@?$C@H@@QEAAXXZ"
// GNU: define weak_odr dso_local dllexport void @_ZN1CIiE22not_to_be_instantiatedEv
