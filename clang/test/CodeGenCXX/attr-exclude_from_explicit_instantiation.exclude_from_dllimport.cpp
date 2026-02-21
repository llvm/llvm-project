// RUN: %clang_cc1 -triple x86_64-win32 -fms-extensions -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=MSC --implicit-check-not=to_be_ --implicit-check-not=dllimport
// RUN: %clang_cc1 -triple x86_64-mingw                 -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllimport
// RUN: %clang_cc1 -triple x86_64-cygwin                -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefixes=GNU --implicit-check-not=to_be_ --implicit-check-not=dllimport

// Test that __declspec(dllimport) doesn't instantiate entities marked with
// the exclude_from_explicit_instantiation attribute unless marked as dllimport explicitly.

// Silence --implicit-check-not=dllimport.
// MSC: ModuleID = {{.*}}exclude_from_dllimport.cpp
// MSC: source_filename = {{.*}}exclude_from_dllimport.cpp
// GNU: ModuleID = {{.*}}exclude_from_dllimport.cpp
// GNU: source_filename = {{.*}}exclude_from_dllimport.cpp

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct C {
  // This will be instantiated explicitly as an imported function because it
  // inherits dllimport from the class instantiation.
  void to_be_imported();

  // This will be instantiated implicitly as an imported function because it is
  // marked as dllimport explicitly.
  EXCLUDE_ATTR __declspec(dllimport) void to_be_imported_explicitly();

  // This will be instantiated implicitly but won't be imported.
  EXCLUDE_ATTR void not_to_be_imported();

  // This won't be instantiated.
  EXCLUDE_ATTR void not_to_be_instantiated();
};

template <class T> void C<T>::to_be_imported() {}
template <class T> void C<T>::not_to_be_imported() {}
template <class T> void C<T>::not_to_be_instantiated() {}

// Attach the attribute to class template declaration instead of instantiation declaration.
template <class T>
struct __declspec(dllimport) D {
  // This will be imported if and only if no explicit instantiations are provided.
  EXCLUDE_ATTR void to_be_imported_iff_no_explicit_instantiation();
};

template <class T> void D<T>::to_be_imported_iff_no_explicit_instantiation() {}

// Interaction with VTables.
template <class T>
struct E {
  // For the MSVC ABI: this constructor causes implicit instantiation of
  // the VTable, which triggers instantiating all virtual member
  // functions regardless `exclude_from_explicit_instantiation`.
  // For the Itanium ABI: Emitting the VTable is suppressed by implicit
  // instantiation declaration so virtual member functions won't be instantiated.
  EXCLUDE_ATTR explicit E(int);

  // This constructor doesn't trigger the instantiation of the VTable.
  // In this case, declaration of virtual member functions are absent too.
  explicit E(long);

  // The body of this shouldn't be emitted since instantiation is suppressed
  // by the explicit instantiation declaration.
  virtual void to_be_imported();

  // The body of this should be emitted if the VTable is instantiated, even if
  // the instantiation of this class template is declared with dllimport.
  EXCLUDE_ATTR virtual void to_be_instantiated();

  // The body of this shouldn't be emitted since that comes from an external DLL.
  EXCLUDE_ATTR __declspec(dllimport) virtual void to_be_imported_explicitly();

};

template <class T> E<T>::E(int) {}
template <class T> E<T>::E(long) {}
template <class T> void E<T>::to_be_imported() {}
template <class T> void E<T>::to_be_instantiated() {}

// MSC: $"?not_to_be_imported@?$C@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_imported_iff_no_explicit_instantiation@?$D@H@@QEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$E@H@@UEAAXXZ" = comdat any
// MSC: $"?to_be_instantiated@?$E@I@@UEAAXXZ" = comdat any
// GNU: $_ZN1CIiE18not_to_be_importedEv = comdat any
// GNU: $_ZN1DIiE44to_be_imported_iff_no_explicit_instantiationEv = comdat any
// GNU: @_ZTV1EIiE = external dllimport unnamed_addr
// GNU: @_ZTV1EIjE = external unnamed_addr

// MSC: @0 = private unnamed_addr constant {{.*}}, comdat($"??_S?$E@H@@6B@")
// MSC: @1 = private unnamed_addr constant {{.*}}, comdat($"??_7?$E@I@@6B@")
// MSC: @"??_S?$E@H@@6B@" =
// MSC: @"??_7?$E@I@@6B@" =

extern template struct __declspec(dllimport) C<int>;

extern template struct D<int>; // No dllimport here.
// Don't provide explicit instantiation for D<unsigned>.

extern template struct __declspec(dllimport) E<int>;      // $E@H, 1EIiE
extern template struct E<unsigned>;                       // $E@I, 1EIjE
extern template struct __declspec(dllimport) E<long int>; // $E@J, 1EIlE
extern template struct E<unsigned long int>;              // $E@K, 1EImE

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

  D<int> di;

  // MSC: call void @"?to_be_imported_iff_no_explicit_instantiation@?$D@H@@QEAAXXZ"
  // GNU: call void @_ZN1DIiE44to_be_imported_iff_no_explicit_instantiationEv
  di.to_be_imported_iff_no_explicit_instantiation(); // implicitly instantiated here

  D<unsigned> dj;

  // MSC: call void @"?to_be_imported_iff_no_explicit_instantiation@?$D@I@@QEAAXXZ"
  // GNU: call void @_ZN1DIjE44to_be_imported_iff_no_explicit_instantiationEv
  dj.to_be_imported_iff_no_explicit_instantiation(); // implicitly instantiated here

  E<int> ei{1};

  E<unsigned> ej{1};

  E<long int> el{1L};

  E<unsigned long int> em{1L};
}

// MSC: declare dllimport void @"?to_be_imported@?$C@H@@QEAAXXZ"
// GNU: declare dllimport void @_ZN1CIiE14to_be_importedEv

// MSC: declare dllimport void @"?to_be_imported_explicitly@?$C@H@@QEAAXXZ"
// GNU: declare dllimport void @_ZN1CIiE25to_be_imported_explicitlyEv

// MSC: define linkonce_odr dso_local void @"?not_to_be_imported@?$C@H@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1CIiE18not_to_be_importedEv

// MSC: define linkonce_odr dso_local void @"?to_be_imported_iff_no_explicit_instantiation@?$D@H@@QEAAXXZ"
// MSC: declare dllimport void @"?to_be_imported_iff_no_explicit_instantiation@?$D@I@@QEAAXXZ"
// GNU: define linkonce_odr dso_local void @_ZN1DIiE44to_be_imported_iff_no_explicit_instantiationEv
// GNU: declare dllimport void @_ZN1DIjE44to_be_imported_iff_no_explicit_instantiationEv

// MSC: declare dllimport noundef ptr @"??0?$E@J@@QEAA@J@Z"
// MSC: declare dso_local noundef ptr @"??0?$E@K@@QEAA@J@Z"
// GNU: define linkonce_odr dso_local void @_ZN1EIiEC1Ei
// GNU: define linkonce_odr dso_local void @_ZN1EIjEC1Ei
// GNU: declare dllimport void @_ZN1EIlEC1El
// GNU: declare dso_local void @_ZN1EImEC1El
// GNU: define linkonce_odr dso_local void @_ZN1EIiEC2Ei
// GNU: define linkonce_odr dso_local void @_ZN1EIjEC2Ei

// MSC: declare dllimport void @"?to_be_imported@?$E@H@@UEAAXXZ"
// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$E@H@@UEAAXXZ"
// MSC: declare dllimport void @"?to_be_imported_explicitly@?$E@H@@UEAAXXZ"

// MSC: declare dso_local void @"?to_be_imported@?$E@I@@UEAAXXZ"
// MSC: define linkonce_odr dso_local void @"?to_be_instantiated@?$E@I@@UEAAXXZ"
// MSC: declare dllimport void @"?to_be_imported_explicitly@?$E@I@@UEAAXXZ"
