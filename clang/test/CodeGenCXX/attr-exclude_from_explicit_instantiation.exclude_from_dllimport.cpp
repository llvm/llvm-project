// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=MSC --implicit-check-not=to_be_ --implicit-check-not=dllimport
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllimport
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllimport

// Test that __declspec(dllimport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute unless marked as dllimport explicitly.

// MSC: ModuleID = {{.*}}exclude_from_dllimport.cpp
// MSC: source_filename = {{.*}}exclude_from_dllimport.cpp
// GNU: ModuleID = {{.*}}exclude_from_dllimport.cpp
// GNU: source_filename = {{.*}}exclude_from_dllimport.cpp

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct C {
  // This will be instantiated explicitly as an imported function because it
  // inherits dllimport from the class instantiation.
  void to_be_imported() noexcept;

  // This will be instantiated implicitly as an imported function because it is
  // marked as dllimport explicitly.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION __declspec(dllimport) void to_be_imported_explicitly() noexcept;

  // This will be instantiated implicitly but won't be imported.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void not_to_be_imported() noexcept;

  // This won't be instantiated.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void not_to_be_instantiated() noexcept;
};

template <class T> void C<T>::to_be_imported() noexcept {}
template <class T> void C<T>::not_to_be_imported() noexcept {}
template <class T> void C<T>::not_to_be_instantiated() noexcept {}

// Attach the attribute to class template declaration instead of instantiation declaration.
template <class T>
struct __declspec(dllimport) D {
  // This will be imported by the class-level attribute.
  void to_be_imported() noexcept;

  // This also should be imported by the class-level attribute but currently not.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void also_to_be_imported() noexcept;
};

template <class T> void D<T>::to_be_imported() noexcept {}
template <class T> void D<T>::also_to_be_imported() noexcept {}

// MSC: $"?not_to_be_imported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?also_to_be_imported@?$D@H@@QEAAXXZ" = comdat any
// GNU: $_ZN1CIiE18not_to_be_importedEv = comdat any
// GNU: $_ZN1DIiE19also_to_be_importedEv = comdat any

extern template struct __declspec(dllimport) C<int>;

extern template struct D<int>;

void use() {
  C<int> c;

  // MSC: call void @"?to_be_imported@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE14to_be_importedEv
  c.to_be_imported();

  // MSC: call void @"?to_be_imported_explicitly@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE25to_be_imported_explicitlyEv
  c.to_be_imported_explicitly(); // implicitly instantiated here

  // MSC: call void @"?not_to_be_imported@?$C@H@@QEAAXXZ"
  // GNU: call void @_ZN1CIiE18not_to_be_importedEv
  c.not_to_be_imported(); // implicitly instantiated here

  D<int> d;

  // MSC: call void @"?to_be_imported@?$D@H@@QEAAXXZ"
  // GNU: call void @_ZN1DIiE14to_be_importedEv
  d.to_be_imported(); // implicitly instantiated here

  // MSC: call void @"?also_to_be_imported@?$D@H@@QEAAXXZ"
  // GNU: call void @_ZN1DIiE19also_to_be_importedEv
  d.also_to_be_imported(); // implicitly instantiated here
}

// MSC: declare dllimport void @"?to_be_imported@?$C@H@@QEAAXXZ"
// GNU: declare dllimport void @_ZN1CIiE14to_be_importedEv

// MSC: declare dllimport void @"?to_be_imported_explicitly@?$C@H@@QEAAXXZ"
// GNU: declare dllimport void @_ZN1CIiE25to_be_imported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?not_to_be_imported@?$C@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1CIiE18not_to_be_importedEv

// MSC: declare dllimport void @"?to_be_imported@?$D@H@@QEAAXXZ"
// GNU: declare dllimport void @_ZN1DIiE14to_be_importedEv

// MSC: define linkonce_odr dso_local void @"?also_to_be_imported@?$D@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1DIiE19also_to_be_importedEv
