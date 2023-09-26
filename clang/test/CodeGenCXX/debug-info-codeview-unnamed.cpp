// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=limited -S -emit-llvm -std=c++11 -o - %s | FileCheck --check-prefix LINUX %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=limited -gcodeview -S -emit-llvm -std=c++11 -o - %s | FileCheck --check-prefix MSVC %s

int main(int argc, char* argv[], char* arge[]) {
  //
  // LINUX-DAG:      [[TYPE_OF_ONE:![0-9]+]] = distinct !DICompositeType(
  // LINUX-DAG-SAME:     tag: DW_TAG_structure_type
  // LINUX-DAG-SAME-NOT:      name:
  // LINUX-DAG-SAME-NOT:      identifier:
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       [[TYPE_OF_ONE:![0-9]+]] = distinct !DICompositeType
  // MSVC-DAG-SAME:      tag: DW_TAG_structure_type
  // MSVC-DAG-SAME:      name: "<unnamed-type-one>"
  // MSVC-DAG-SAME:      identifier: ".?AU<unnamed-type-one>@?1??main@@9@"
  // MSVC-DAG-SAME:      )


  //
  // LINUX-DAG:      [[TYPE_OF_TWO:![0-9]+]] = distinct !DICompositeType(
  // LINUX-DAG-SAME:     tag: DW_TAG_structure_type
  // LINUX-DAG-SAME-NOT:      name:
  // LINUX-DAG-SAME-NOT:      identifier:
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       [[TYPE_OF_TWO:![0-9]+]] = distinct !DICompositeType
  // MSVC-DAG-SAME:      tag: DW_TAG_structure_type
  // MSVC-DAG-SAME:      name: "<unnamed-type-two>"
  // MSVC-DAG-SAME:      identifier: ".?AU<unnamed-type-two>@?2??main@@9@"
  // MSVC-DAG-SAME:      )


  //
  // LINUX-DAG:      [[TYPE_OF_THREE:![0-9]+]] = distinct !DICompositeType(
  // LINUX-DAG-SAME:     tag: DW_TAG_structure_type
  // LINUX-DAG-SAME:     name: "named"
  // LINUX-DAG-SAME-NOT:      identifier:
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       [[TYPE_OF_THREE:![0-9]+]] = distinct !DICompositeType
  // MSVC-DAG-SAME:      tag: DW_TAG_structure_type
  // MSVC-DAG-SAME:      name: "named"
  // MSVC-DAG-SAME:      identifier: ".?AUnamed@?1??main@@9@"
  // MSVC-DAG-SAME:      )

  //
  // LINUX-DAG:      [[TYPE_OF_FOUR:![0-9]+]] = distinct !DICompositeType(
  // LINUX-DAG-SAME:     tag: DW_TAG_class_type
  // LINUX-DAG-SAME-NOT:      name:
  // LINUX-DAG-SAME-NOT:      identifier:
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       [[TYPE_OF_FOUR:![0-9]+]] = distinct !DICompositeType
  // MSVC-DAG-SAME:      tag: DW_TAG_class_type
  // MSVC-DAG-SAME:      name: "<lambda_0>"
  // MSVC-DAG-SAME:      identifier: ".?AV<lambda_0>@?0??main@@9@"
  // MSVC-DAG-SAME:      )


  // In CodeView, the LF_MFUNCTION entry for "bar()" refers to the forward
  // reference of the unnamed struct. Visual Studio requires a unique
  // identifier to match the LF_STRUCTURE forward reference to the definition.
  //
  struct { void bar() {} } one;
  //
  // LINUX-DAG:      !{{[0-9]+}} = !DILocalVariable(name: "one"
  // LINUX-DAG-SAME:     type: [[TYPE_OF_ONE]]
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       !{{[0-9]+}} = !DILocalVariable(name: "one"
  // MSVC-DAG-SAME:      type: [[TYPE_OF_ONE]]
  // MSVC-DAG-SAME:      )


  // In CodeView, the LF_POINTER entry for "ptr2unnamed" refers to the forward
  // reference of the unnamed struct. Visual Studio requires a unique
  // identifier to match the LF_STRUCTURE forward reference to the definition.
  //
  struct { int bar; } two = { 42 };
  int decltype(two)::*ptr2unnamed = &decltype(two)::bar;
  //
  // LINUX-DAG:      !{{[0-9]+}} = !DILocalVariable(name: "two"
  // LINUX-DAG-SAME:     type: [[TYPE_OF_TWO]]
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       !{{[0-9]+}} = !DILocalVariable(name: "two"
  // MSVC-DAG-SAME:      type: [[TYPE_OF_TWO]]
  // MSVC-DAG-SAME:      )


  // In DWARF, named structures which are not externally visibile do not
  // require an identifier.  In CodeView, named structures are given an
  // identifier.
  //
  struct named { int bar; int named::* p2mem; } three = { 42, &named::bar };
  //
  // LINUX-DAG:      !{{[0-9]+}} = !DILocalVariable(name: "three"
  // LINUX-DAG-SAME:     type: [[TYPE_OF_THREE]]
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       !{{[0-9]+}} = !DILocalVariable(name: "three"
  // MSVC-DAG-SAME:      type: [[TYPE_OF_THREE]]
  // MSVC-DAG-SAME:      )


  // In CodeView, the LF_MFUNCTION entry for the lambda "operator()" routine
  // refers to the forward reference of the unnamed LF_CLASS for the lambda.
  // Visual Studio requires a unique identifier to match the forward reference
  // of the LF_CLASS to its definition.
  //
  auto four = [argc](int i) -> int { return argc == i ? 1 : 0; };
  //
  // LINUX-DAG:      !{{[0-9]+}} = !DILocalVariable(name: "four"
  // LINUX-DAG-SAME:     type: [[TYPE_OF_FOUR]]
  // LINUX-DAG-SAME:     )
  //
  // MSVC-DAG:       !{{[0-9]+}} = !DILocalVariable(name: "four"
  // MSVC-DAG-SAME:      type: [[TYPE_OF_FOUR]]
  // MSVC-DAG-SAME:      )

  return 0;
}
