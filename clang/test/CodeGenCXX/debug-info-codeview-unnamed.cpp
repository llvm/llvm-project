// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=limited -emit-llvm -std=c++11 -o - %s | FileCheck --check-prefix LINUX %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -std=c++11 -o - %s | FileCheck --check-prefix MSVC %s

int main(int argc, char* argv[], char* arge[]) {
  //
  // In CodeView, the LF_MFUNCTION entry for "bar()" refers to the forward
  // reference of the unnamed struct. Visual Studio requires a unique
  // identifier to match the LF_STRUCTURE forward reference to the definition.
  //
  struct { void bar() {} } one;
  //
  // LINUX:      !{{[0-9]+}} = !DILocalVariable(name: "one"
  // LINUX-SAME:     type: [[TYPE_OF_ONE:![0-9]+]]
  // LINUX-SAME:     )
  // LINUX:      [[TYPE_OF_ONE]] = distinct !DICompositeType(
  // LINUX-SAME:     tag: DW_TAG_structure_type
  // LINUX-NOT:      name:
  // LINUX-NOT:      identifier:
  // LINUX-SAME:     )
  //
  // MSVC:       !{{[0-9]+}} = !DILocalVariable(name: "one"
  // MSVC-SAME:      type: [[TYPE_OF_ONE:![0-9]+]]
  // MSVC-SAME:      )
  // MSVC:       [[TYPE_OF_ONE]] = distinct !DICompositeType
  // MSVC-SAME:      tag: DW_TAG_structure_type
  // MSVC-SAME:      name: "<unnamed-type-one>"
  // MSVC-SAME:      identifier: ".?AU<unnamed-type-one>@?1??main@@9@"
  // MSVC-SAME:      )


  // In CodeView, the LF_POINTER entry for "ptr2unnamed" refers to the forward
  // reference of the unnamed struct. Visual Studio requires a unique
  // identifier to match the LF_STRUCTURE forward reference to the definition.
  //
  struct { int bar; } two = { 42 };
  int decltype(two)::*ptr2unnamed = &decltype(two)::bar;
  //
  // LINUX:      !{{[0-9]+}} = !DILocalVariable(name: "two"
  // LINUX-SAME:     type: [[TYPE_OF_TWO:![0-9]+]]
  // LINUX-SAME:     )
  // LINUX:      [[TYPE_OF_TWO]] = distinct !DICompositeType(
  // LINUX-SAME:     tag: DW_TAG_structure_type
  // LINUX-NOT:      name:
  // LINUX-NOT:      identifier:
  // LINUX-SAME:     )
  //
  // MSVC:       !{{[0-9]+}} = !DILocalVariable(name: "two"
  // MSVC-SAME:      type: [[TYPE_OF_TWO:![0-9]+]]
  // MSVC-SAME:      )
  // MSVC:       [[TYPE_OF_TWO]] = distinct !DICompositeType
  // MSVC-SAME:      tag: DW_TAG_structure_type
  // MSVC-SAME:      name: "<unnamed-type-two>"
  // MSVC-SAME:      identifier: ".?AU<unnamed-type-two>@?2??main@@9@"
  // MSVC-SAME:      )


  // In DWARF, named structures which are not externally visibile do not
  // require an identifier.  In CodeView, named structures are given an
  // identifier.
  //
  struct named { int bar; int named::* p2mem; } three = { 42, &named::bar };
  //
  // LINUX:      !{{[0-9]+}} = !DILocalVariable(name: "three"
  // LINUX-SAME:     type: [[TYPE_OF_THREE:![0-9]+]]
  // LINUX-SAME:     )
  // LINUX:      [[TYPE_OF_THREE]] = distinct !DICompositeType(
  // LINUX-SAME:     tag: DW_TAG_structure_type
  // LINUX-SAME:     name: "named"
  // LINUX-NOT:      identifier:
  // LINUX-SAME:     )
  //
  // MSVC:       !{{[0-9]+}} = !DILocalVariable(name: "three"
  // MSVC-SAME:      type: [[TYPE_OF_THREE:![0-9]+]]
  // MSVC-SAME:      )
  // MSVC:       [[TYPE_OF_THREE]] = distinct !DICompositeType
  // MSVC-SAME:      tag: DW_TAG_structure_type
  // MSVC-SAME:      name: "named"
  // MSVC-SAME:      identifier: ".?AUnamed@?1??main@@9@"
  // MSVC-SAME:      )


  // In CodeView, the LF_MFUNCTION entry for the lambda "operator()" routine
  // refers to the forward reference of the unnamed LF_CLASS for the lambda.
  // Visual Studio requires a unique identifier to match the forward reference
  // of the LF_CLASS to its definition.
  //
  auto four = [argc](int i) -> int { return argc == i ? 1 : 0; };
  //
  // LINUX:      !{{[0-9]+}} = !DILocalVariable(name: "four"
  // LINUX-SAME:     type: [[TYPE_OF_FOUR:![0-9]+]]
  // LINUX-SAME:     )
  // LINUX:      [[TYPE_OF_FOUR]] = distinct !DICompositeType(
  // LINUX-SAME:     tag: DW_TAG_class_type
  // LINUX-NOT:      name:
  // LINUX-NOT:      identifier:
  // LINUX-SAME:     )
  //
  // MSVC:       !{{[0-9]+}} = !DILocalVariable(name: "four"
  // MSVC-SAME:      type: [[TYPE_OF_FOUR:![0-9]+]]
  // MSVC-SAME:      )
  // MSVC:       [[TYPE_OF_FOUR]] = distinct !DICompositeType
  // MSVC-SAME:      tag: DW_TAG_class_type
  // MSVC-SAME:      name: "<lambda_0>"
  // MSVC-SAME:      identifier: ".?AV<lambda_0>@?0??main@@9@"
  // MSVC-SAME:      )

  return 0;
}
