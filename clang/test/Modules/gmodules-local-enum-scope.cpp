// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -o %t/header.pch %t/header.h
// RUN: %clang_cc1 -std=c++11 -debug-info-kind=limited -dwarf-ext-refs \
// RUN:     -include-pch %t/header.pch %t/main.cpp -emit-llvm -o - \
// RUN:   | FileCheck %s

// Test that type nodes for function-local enums are emitted in their lexical blocks
// when deserialized from a PCH with -dwarf-ext-refs, including when the enum's declaration
// and definition are separate.

//--- header.h
inline void f() {
  {
    enum E : int;
    E e{};
    enum E : int { A };
  }
}

inline void g() {
  {
    enum F : int { B };
    F x{};
  }
}

//--- main.cpp
void use() {
  f();
  g();
}

// CHECK:       !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", scope: ![[E_SCOPE:[0-9]+]]
// CHECK-SAME:                   flags: DIFlagFwdDecl
// CHECK:       ![[E_SCOPE]] = distinct !DILexicalBlock(

// CHECK:       !DICompositeType(tag: DW_TAG_enumeration_type, name: "F", scope: ![[F_SCOPE:[0-9]+]]
// CHECK-SAME:                   flags: DIFlagFwdDecl
// CHECK:       ![[F_SCOPE]] = distinct !DILexicalBlock(
