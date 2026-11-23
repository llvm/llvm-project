// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -debugger-tuning=lldb %s | FileCheck %s --check-prefixes=COMMON,LLDB
// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -debugger-tuning=gdb %s | FileCheck %s --check-prefixes=COMMON,GDB

template <typename T>
struct Foo;

typedef Foo<int> BarInt;
typedef Foo<double> BarDouble;

template <typename T>
using Bar = Foo<T>;

template <typename T>
struct [[clang::preferred_name(BarInt),
         clang::preferred_name(BarDouble),
         clang::preferred_name(Bar<short>),
         clang::preferred_name(Bar<char>)]] Foo{
         };

int main() {
    Foo<int> varInt;

// COMMON: !DILocalVariable(name: "varInt", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_INT_TY:[0-9]+]])
// LLDB:   ![[BAR_INT_TY]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarInt", file: ![[#]], line: [[#]], baseType: ![[BAR_INT_BASE:[0-9]+]])
// LLDB:   ![[BAR_INT_BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<int>", file: ![[#]], line: [[#]], size: [[#]]
// GDB:    ![[BAR_INT_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<int>", file: ![[#]], line: [[#]], size: [[#]]

    Foo<double> varDouble;

// COMMON: !DILocalVariable(name: "varDouble", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_DOUBLE_TY:[0-9]+]])
// LLDB:   ![[BAR_DOUBLE_TY]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarDouble", file: ![[#]], line: [[#]], baseType: ![[BAR_DOUBLE_BASE:[0-9]+]])
// LLDB:   ![[BAR_DOUBLE_BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<double>"
// GDB:    ![[BAR_DOUBLE_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<double>"

    Foo<short> varShort;

// COMMON: !DILocalVariable(name: "varShort", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_SHORT_TY:[0-9]+]])
// LLDB:   ![[BAR_SHORT_TY]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<short>", file: ![[#]], line: [[#]], baseType: ![[BAR_SHORT_BASE:[0-9]+]])
// LLDB:   ![[BAR_SHORT_BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<short>"
// GDB:    ![[BAR_SHORT_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<short>"

    Foo<char> varChar;

// COMMON: !DILocalVariable(name: "varChar", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_CHAR_TY:[0-9]+]])
// LLDB:   ![[BAR_CHAR_TY]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<char>", file: ![[#]], line: [[#]], baseType: ![[BAR_CHAR_BASE:[0-9]+]])
// LLDB:   ![[BAR_CHAR_BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<char>"
// GDB:    ![[BAR_CHAR_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<char>"

    Foo<Foo<int>> varFooInt;

// COMMON:      !DILocalVariable(name: "varFooInt", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[FOO_BAR_INT_TY:[0-9]+]])
// COMMON:      ![[FOO_BAR_INT_TY]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<Foo<int> >"
// COMMON-SAME: templateParams: ![[PARAM:[0-9]+]]
// COMMON:      ![[PARAM]] = !{![[TEMPL_TYPE_PARAM:[0-9]+]]}
// GDB:         ![[TEMPL_TYPE_PARAM]] = !DITemplateTypeParameter(name: "T", type: ![[BAR_INT_TY]])
// LLDB:        ![[TEMPL_TYPE_PARAM]] = !DITemplateTypeParameter(name: "T", type: ![[BAR_INT_TY]])

    BarInt barInt;

// LLDB:   !DILocalVariable(name: "barInt", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_INT_TY]])
// GDB:    !DILocalVariable(name: "barInt", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_INT_TYPEDEF:[0-9]+]])
// GDB:    ![[BAR_INT_TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarInt"

    BarDouble barDouble;

// LLDB:   !DILocalVariable(name: "barDouble", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_DOUBLE_TY]])
// GDB:    !DILocalVariable(name: "barDouble", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_DOUBLE_TYPEDEF:[0-9]+]])
// GDB:    ![[BAR_DOUBLE_TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BarDouble"

    Bar<short> barShort;

// LLDB:   !DILocalVariable(name: "barShort", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_SHORT_TY_2:[0-9]+]])
// GDB:    !DILocalVariable(name: "barShort", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_SHORT_TYPEDEF:[0-9]+]])
// GDB:    ![[BAR_SHORT_TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<short>"

    Bar<char> barChar;

// LLDB:   ![[BAR_SHORT_TY_2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<short>", file: ![[#]], line: [[#]], baseType: ![[BAR_SHORT_TY]])

// LLDB:   !DILocalVariable(name: "barChar", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_CHAR_TY_2:[0-9]+]])
// GDB:    !DILocalVariable(name: "barChar", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[BAR_CHAR_TYPEDEF:[0-9]+]])
// GDB:    ![[BAR_CHAR_TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<char>"

// LLDB:   ![[BAR_CHAR_TY_2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<char>", file: ![[#]], line: [[#]], baseType: ![[BAR_CHAR_TY]])

    return 0;
}


