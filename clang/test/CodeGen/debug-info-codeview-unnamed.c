// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck --check-prefix LINUX %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=limited -gcodeview -S -emit-llvm -o - %s | FileCheck --check-prefix MSVC %s

int main(int argc, char* argv[], char* arge[]) {

  // In both DWARF and CodeView, an unnamed C structure type will generate a
  // DICompositeType without a name or identifier attribute;
  //
  struct { int bar; } one = {42};
  //
  // LINUX:      [[TYPE_OF_ONE:![0-9]+]] = distinct !DICompositeType(
  // LINUX-SAME:     tag: DW_TAG_structure_type
  // LINUX-NOT:      name:
  // LINUX-NOT:      identifier:
  // LINUX-SAME:     )
  // LINUX:      !{{[0-9]+}} = !DILocalVariable(name: "one"
  // LINUX-SAME:     type: [[TYPE_OF_ONE]]
  // LINUX-SAME:     )
  //
  // MSVC:       [[TYPE_OF_ONE:![0-9]+]] = distinct !DICompositeType
  // MSVC-SAME:      tag: DW_TAG_structure_type
  // MSVC-NOT:       name:
  // MSVC-NOT:       identifier:
  // MSVC-SAME:      )
  // MSVC:       !{{[0-9]+}} = !DILocalVariable(name: "one"
  // MSVC-SAME:      type: [[TYPE_OF_ONE]]
  // MSVC-SAME:      )

  return 0;
}
