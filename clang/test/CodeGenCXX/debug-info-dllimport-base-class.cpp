// RUN: %clang_cc1 -triple i386-pc-windows -emit-llvm -gcodeview -debug-info-kind=limited -fms-compatibility %s -x c++ -o - | FileCheck %s

// Ensure we emit debug info for the full definition of base classes that will
// be imported from a DLL.  Otherwise, the debugger wouldn't be able to show the
// members.

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "OutOfLineCtor",
// CHECK-SAME:             DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ImportedBase",
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ImportedMethod",
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

struct OutOfLineCtor {
  OutOfLineCtor();
  virtual void Foo();
};

struct __declspec(dllimport) ImportedBase {
  ImportedBase();
  virtual void Foo();
};

struct DerivedFromImported : public ImportedBase {};

struct ImportedMethod {
  ImportedMethod();
  virtual void Foo();
  static void __declspec(dllimport) create();
};

int main() {
  OutOfLineCtor o;
  DerivedFromImported d;
  ImportedMethod m;
}
