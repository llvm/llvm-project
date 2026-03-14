/// For a class that has a vtable and typeinfo symbol for RTTI, if a user marks
/// either:
///
///   (a) The entire class as dllexport (dllimport)
///   (b) Any non-inline method of the class as dllexport (dllimport)
///
/// then Clang must export the vtable and typeinfo symbol from the TU where they
/// are defined (the TU containing the definition of the Itanium C++ ABI "key
/// function") and must import them in other modules where they are referenced.

// RUN: %clang_cc1 -I%S -fdeclspec -triple x86_64-unknown-windows-itanium -emit-llvm -o - %s -fhalf-no-semantic-interposition \
// RUN:   | FileCheck %s -check-prefix=WI
// RUN: %clang_cc1 -I%S -fdeclspec -triple x86_64-scei-windows-itanium    -emit-llvm -o - %s -fhalf-no-semantic-interposition \
// RUN:   | FileCheck %s --check-prefixes=PS
// RUN: %clang_cc1 -I%S -fdeclspec -triple x86_64-scei-ps4 -emit-llvm -o - %s -fhalf-no-semantic-interposition \
// RUN:   | FileCheck %s --check-prefixes=PS
// RUN: %clang_cc1 -I%S -fdeclspec -triple x86_64-sie-ps5  -emit-llvm -o - %s -fhalf-no-semantic-interposition \
// RUN:   | FileCheck %s --check-prefixes=PS

#include <typeinfo>

/// Case (a) -- Import Aspect
/// The entire class is imported. The typeinfo symbol must also be imported, but
/// the vtable will not be referenced, and so does not need to be imported.

// PS-DAG: @_ZTI10FullImport = {{.*}}dllimport
// WI-DAG: @_ZTI10FullImport = external dllimport constant ptr
struct __declspec(dllimport) FullImport {
  virtual void inlineFunc() const {}
  virtual void key();
  virtual void func();
};

/// 'FullImport::key()' is the key function, so the vtable and typeinfo symbol
/// of 'FullImport' will be defined in the TU that contains the definition of
/// 'key()' (and they must be exported from there).
void FullImportTest() { typeid(FullImport).name(); }

/// Case (a) -- Export Aspect
/// The entire class is exported. The vtable and typeinfo symbols must also be
/// exported.

// PS-DAG: @_ZTV10FullExport = {{.*}}dllexport
// WI-DAG: @_ZTV10FullExport = {{.*}}dllexport
// PS-DAG: @_ZTI10FullExport = {{.*}}dllexport
// WI-DAG: @_ZTI10FullExport = dso_local dllexport constant {
struct __declspec(dllexport) FullExport {
  virtual void inlineFunc() const {}
  virtual void key();
  virtual void func();
};

/// This is the key function of the class 'FullExport', so the vtable and
/// typeinfo symbols of 'FullExport' will be defined in this TU, and so they
/// must be exported from this TU.
void FullExport::key() { typeid(FullExport).name(); }

/// Case (b) -- Import Aspect
/// The class as a whole is not imported, but a non-inline method of the class
/// is, so the vtable and typeinfo symbol must be imported.

// PS-DAG: @_ZTV10PartImport = {{.*}}dllimport
// WI-DAG: @_ZTV10PartImport = external dso_local unnamed_addr constant {
// PS-DAG: @_ZTI10PartImport = {{.*}}dllimport
// WI-DAG: @_ZTI10PartImport = external dso_local constant ptr
struct PartImport {
  virtual void inlineFunc() const {}
  virtual void key();
  __declspec(dllimport) virtual void func();
};

/// 'PartImport::key()' is the key function, so the vtable and typeinfo symbol
/// of 'PartImport' will be defined in the TU that contains the definition of
/// 'key()' (and they must be exported from there). Here, we will reference the
/// vtable and typeinfo symbol, so we must also import them.
void PartImportTest() {
  PartImport f;
  typeid(PartImport).name();
}

/// Case (b) -- Export Aspect
/// The class as a whole is not exported, but a non-inline method of the class
/// is, so the vtable and typeinfo symbol must be exported.

// PS-DAG: @_ZTV10PartExport = {{.*}}dllexport
// WI-DAG: @_ZTV10PartExport = dso_local unnamed_addr constant {
// PS-DAG: @_ZTI10PartExport = {{.*}}dllexport
// WI-DAG: @_ZTI10PartExport = dso_local constant {
struct PartExport {
  virtual void inlineFunc() const {}
  virtual void key();
  __declspec(dllexport) virtual void func();
};

/// This is the key function of the class 'PartExport', so the vtable and
/// typeinfo symbol of 'PartExport' will be defined in this TU, and so they must
/// be exported from this TU.
void PartExport::key() { typeid(PartExport).name(); }

/// Case (b) -- Export Aspect
/// The class as a whole is not exported, but the constructor of the class
/// is, so the vtable and typeinfo symbol must be exported.

// PS-DAG: @_ZTV10ConsExport = {{.*}}dllexport
// WI-DAG: @_ZTV10ConsExport = dso_local unnamed_addr constant {
// PS-DAG: @_ZTI10ConsExport = {{.*}}dllexport
// WI-DAG: @_ZTI10ConsExport = dso_local constant {
struct ConsExport {
  __declspec(dllexport) ConsExport();
  virtual void key();
};

ConsExport::ConsExport() {}
void ConsExport::key() { typeid(ConsExport).name(); }
