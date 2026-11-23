// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -debugger-tuning=lldb %s | FileCheck %s --check-prefixes=COMMON,LLDB
// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -debugger-tuning=gdb %s | FileCheck %s --check-prefixes=COMMON,GDB

template <typename T>
struct Foo;

typedef Foo<int> BarIntBase;
typedef BarIntBase BarIntTmp;
typedef BarIntTmp BarInt;

template <typename T>
using BarBase = Foo<T>;

template <typename T>
using BarTmp = BarBase<T>;

template <typename T>
using Bar = BarTmp<T>;

template <typename T>
struct [[clang::preferred_name(BarInt),
         clang::preferred_name(Bar<char>)]] Foo{
         };

int main() {
    Foo<int> varInt;

// COMMON: !DILocalVariable(name: "varInt", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_INT_TY:[0-9]+]])
// LLDB:   ![[BAR_INT_TY]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarInt", file: ![[#]], line: [[#]], baseType: ![[BAR_INT_TMP:[0-9]+]])
// LLDB:   ![[BAR_INT_TMP]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarIntTmp", file: ![[#]], line: [[#]], baseType: ![[BAR_INT_BASE:[0-9]+]])
// LLDB:   ![[BAR_INT_BASE]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarIntBase", file: ![[#]], line: [[#]], baseType: ![[FOO_INT:[0-9]+]])
// LLDB:   ![[FOO_INT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<int>", file: ![[#]], line: [[#]], size: [[#]]
// GDB:    ![[BAR_INT_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<int>", file: ![[#]], line: [[#]], size: [[#]]

    Foo<char> varChar;

// COMMON: !DILocalVariable(name: "varChar", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_CHAR_TY:[0-9]+]])
// LLDB:   ![[BAR_CHAR_TY]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<char>", file: ![[#]], line: [[#]], baseType: ![[BAR_CHAR_TMP:[0-9]+]])
// LLDB:   ![[BAR_CHAR_TMP]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarTmp<char>", file: ![[#]], line: [[#]], baseType: ![[BAR_CHAR_BASE:[0-9]+]])
// LLDB:   ![[BAR_CHAR_BASE]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarBase<char>", file: ![[#]], line: [[#]], baseType: ![[FOO_CHAR:[0-9]+]])
// LLDB:   ![[FOO_CHAR]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<char>"
// GDB:    ![[BAR_CHAR_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<char>"

    Foo<Foo<int>> varFooInt;

// COMMON:      !DILocalVariable(name: "varFooInt", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[FOO_BAR_INT_TY:[0-9]+]])
// COMMON:      ![[FOO_BAR_INT_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<Foo<int> >"
// COMMON-SAME: templateParams: ![[PARAM:[0-9]+]]
// COMMON:      ![[PARAM]] = !{![[TEMPL_TYPE_PARAM:[0-9]+]]}
// GDB:         ![[TEMPL_TYPE_PARAM]] = !DITemplateTypeParameter(name: "T", type: ![[BAR_INT_TY]])
// LLDB:        ![[TEMPL_TYPE_PARAM]] = !DITemplateTypeParameter(name: "T", type: ![[BAR_INT_TY]])

    Foo<Foo<char>> varFooChar;

// COMMON:      !DILocalVariable(name: "varFooChar", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[FOO_BAR_CHAR_TY:[0-9]+]])
// COMMON:      ![[FOO_BAR_CHAR_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<Foo<char> >"
// COMMON-SAME: templateParams: ![[CHAR_PARAM:[0-9]+]]
// COMMON:      ![[CHAR_PARAM]] = !{![[CHAR_TEMPL_TYPE_PARAM:[0-9]+]]}
// GDB:         ![[CHAR_TEMPL_TYPE_PARAM]] = !DITemplateTypeParameter(name: "T", type: ![[BAR_CHAR_TY]])
// LLDB:        ![[CHAR_TEMPL_TYPE_PARAM]] = !DITemplateTypeParameter(name: "T", type: ![[BAR_CHAR_TY]])

    return 0;
}


