// For a class that has a vtable (and hence, also has a typeinfo symbol for
// RTTI), if a user marks either:
//
//  (a) the entire class as dllexport (dllimport), or
//  (b) all non-inline virtual methods of the class as dllexport (dllimport)
//
// then Clang must export the vtable and typeinfo symbol from the TU where they
// are defined (the TU containing the definition of the Itanium C++ ABI "key
// function"), and must import them in other modules where they are referenced.
//
// Conversely to point (b), if some (but not all) of the non-inline virtual
// methods of a class are marked as dllexport (dllimport), then the vtable and
// typeinfo symbols must not be exported (imported).  This will result in a
// link-time failure when linking the importing module.  This link-time failure
// is the desired behavior, because the Microsoft toolchain also gets a
// link-time failure in these cases (and since __declspec(dllexport)
// (__declspec(dllimport)) is a Microsoft extension, our intention is to mimic
// that Microsoft behavior).
//
// Side note: It is within the bodies of constructors (and in some cases,
// destructors) that the vtable is explicitly referenced.  In case (a) above,
// where the entire class is exported (imported), then all constructors (among
// other things) are exported (imported).  So for that situation, an importing
// module for a well-formed program will not actually reference the vtable,
// since constructor calls will all be to functions external to that module
// (and imported into it, from the exporting module).  I.e., all vtable
// references will be in that module where the constructor and destructor
// bodies are, therefore, there will not be a need to import the vtable in
// that case.
//
// This test contains 6 test classes:
//   2 for point (a),
//   2 for point (b),
//   and 2 negative tests for the converse of point (b).
//
// The two tests for each of these points are one for importing, and one for
// exporting.

// RUN: %clang_cc1 -no-opaque-pointers -I%S -fdeclspec -triple x86_64-unknown-windows-itanium -emit-llvm -o - %s -fhalf-no-semantic-interposition | FileCheck %s -check-prefix=WI
// RUN: %clang_cc1 -no-opaque-pointers -I%S -fdeclspec -triple x86_64-scei-windows-itanium -emit-llvm -o - %s -fhalf-no-semantic-interposition | FileCheck %s --check-prefixes=PS4,SCEI_WI
// RUN: %clang_cc1 -no-opaque-pointers -I%S -fdeclspec -triple x86_64-scei-ps4 -emit-llvm -o - %s -fhalf-no-semantic-interposition | FileCheck %s --check-prefixes=PS4,SCEI_PS4

#include <typeinfo>

// Case (a) -- Import Aspect
// The entire class is imported.  The typeinfo symbol must also be imported,
// but the vtable will not be referenced, and so does not need to be imported
// (as described in the "Side note", above).
//
// PS4-DAG: @_ZTI10FullImport = {{.*}}dllimport
// WI-DAG: @_ZTI10FullImport = external dllimport constant i8*
struct __declspec(dllimport) FullImport
{
  virtual void getId() {}
  virtual void Bump();
  virtual void Decrement();
};

// 'FullImport::Bump()' is the key function, so the vtable and typeinfo symbol
// of 'FullImport' will be defined in the TU that contains the definition of
// 'Bump()' (and they must be exported from there).
void FullImportTest()
{
  typeid(FullImport).name();
}

///////////////////////////////////////////////////////////////////

// Case (a) -- Export Aspect
// The entire class is exported.  The vtable and typeinfo symbols must also be
// exported,
//
// PS4-DAG: @_ZTV10FullExport ={{.*}}dllexport
// WI-DAG: @_ZTV10FullExport ={{.*}}dllexport
// PS4-DAG: @_ZTI10FullExport ={{.*}}dllexport
// WI-DAG: @_ZTI10FullExport = dso_local dllexport constant {
struct __declspec(dllexport) FullExport // Easy case: Entire class is exported.
{
  virtual void getId() {}
  virtual void Bump();
  virtual void Decrement();
};

// This is the key function of the class 'FullExport', so the vtable and
// typeinfo symbols of 'FullExport' will be defined in this TU, and so they
// must be exported from this TU.
void FullExport::Bump()
{
  typeid(FullExport).name();
}

///////////////////////////////////////////////////////////////////

// Case (b) -- Import Aspect
// The class as a whole is not imported, but all non-inline virtual methods of
// the class are, so the vtable and typeinfo symbol must be imported.
//
// PS4-DAG: @_ZTV9FooImport ={{.*}}dllimport
// WI-DAG:  @_ZTV9FooImport = linkonce_odr dso_local unnamed_addr constant {
// PS4-DAG: @_ZTI9FooImport ={{.*}}dllimport
// WI-DAG:  @_ZTI9FooImport = linkonce_odr dso_local constant {


struct FooImport
{
  virtual void getId() const {}
  __declspec(dllimport) virtual void Bump();
  __declspec(dllimport) virtual void Decrement();
};

// 'FooImport::Bump()' is the key function, so the vtable and typeinfo symbol
// of 'FooImport' will be defined in the TU that contains the definition of
// 'Bump()' (and they must be exported from there).  Here, we will reference
// the vtable and typeinfo symbol, so we must also import them.
void importTest()
{
  typeid(FooImport).name();
}

///////////////////////////////////////////////////////////////////

// Case (b) -- Export Aspect
// The class as a whole is not exported, but all non-inline virtual methods of
// the class are, so the vtable and typeinfo symbol must be exported.
//
// PS4-DAG: @_ZTV9FooExport ={{.*}}dllexport
// WI-DAG:  @_ZTV9FooExport = dso_local unnamed_addr constant {
// PS4-DAG: @_ZTI9FooExport ={{.*}}dllexport
// WI-DAG:  @_ZTI9FooExport = dso_local constant {
struct FooExport
{
  virtual void getId() const {}
  __declspec(dllexport) virtual void Bump();
  __declspec(dllexport) virtual void Decrement();
};

// This is the key function of the class 'FooExport', so the vtable and
// typeinfo symbol of 'FooExport' will be defined in this TU, and so they must
// be exported from this TU.
void FooExport::Bump()
{
  FooImport f;
  typeid(FooExport).name();
}

///////////////////////////////////////////////////////////////////

// The tests below verify that the associated vtable and typeinfo symbols are
// not imported/exported.  These are the converse of case (b).
//
// Note that ultimately, if the module doing the importing calls a constructor
// of the class with the vtable, or makes a reference to the typeinfo symbol of
// the class, then this will result in an unresolved reference (to the vtable
// or typeinfo symbol) when linking the importing module, and thus a link-time
// failure.
//
// Note that with the Microsoft toolchain there will also be a link-time
// failure when linking the module doing the importing.  With the Microsoft
// toolchain, it will be an unresolved reference to the method 'Decrement()'
// of the approriate class, rather than to the vtable or typeinfo symbol of
// the class, because Microsoft defines the vtable and typeinfo symbol (weakly)
// everywhere they are used.

// Converse of case (b) -- Import Aspect
// The class as a whole is not imported, and not all non-inline virtual methods
// are imported, so the vtable and typeinfo symbol are not to be imported.
//
// CHECK-PS4: @_ZTV11FooNoImport = external dso_local unnamed_addr constant {
// CHECK-WI:  @_ZTV11FooNoImport = linkonce_odr dso_local unnamed_addr constant {
// CHECK-PS4: @_ZTI11FooNoImport = external dso_local constant i8*{{$}}
// CHECK-WI:  @_ZTI11FooNoImport = linkonce_odr dso_local constant {
struct FooNoImport
{
  virtual void getId() const {}
  __declspec(dllimport) virtual void Bump();
  virtual void Decrement();     // Not imported.
  int mCounter;
};

void importNegativeTest()
{
  FooNoImport f;
  typeid(FooNoImport).name();
}

///////////////////////////////////////////////////////////////////

// Converse of case (b) -- Export Aspect
// The class as a whole is not exported, and not all non-inline virtual methods
// are exported, so the vtable and typeinfo symbol are not to be exported.
//
// SCEI_PS4-DAG: @_ZTV11FooNoImport = external unnamed_addr constant {
// SCEI_WI-DAG:  @_ZTV11FooNoExport = dso_local unnamed_addr constant {

// WI-DAG:       @_ZTV11FooNoExport = dso_local unnamed_addr constant {
// SCEI_PS4-DAG: @_ZTI11FooNoExport = constant {
// SCEI_WI-DAG:  @_ZTI11FooNoExport = dso_local constant {
// WI-DAG:       @_ZTI11FooNoExport = dso_local constant {
struct FooNoExport
{
  virtual void getId() const {}
  __declspec(dllexport) virtual void Bump();
  virtual void Decrement(); // Not exported.
  int mCounter;
};

void FooNoExport::Bump()
{
  typeid(FooNoExport).name();
}
