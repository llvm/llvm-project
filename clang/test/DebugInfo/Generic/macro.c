// RUN: %clang_cc1 -emit-llvm -debug-info-kind=line-tables-only -debug-info-macro %s -o - "-DC1(x)=( x  + 5 )" -DA -include %S/Inputs/debug-info-macro.h -UC1 | FileCheck -check-prefixes=CHECK,NO_PCH %s 
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=line-directives-only -debug-info-macro %s -o - "-DC1(x)=( x  + 5 )" -DA -include %S/Inputs/debug-info-macro.h -UC1 | FileCheck -check-prefixes=CHECK,NO_PCH %s 
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited          -debug-info-macro %s -o - "-DC1(x)=( x  + 5 )" -DA -include %S/Inputs/debug-info-macro.h -UC1 | FileCheck -check-prefixes=CHECK,NO_PCH %s 
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone       -debug-info-macro %s -o - "-DC1(x)=( x  + 5 )" -DA -include %S/Inputs/debug-info-macro.h -UC1 | FileCheck -check-prefixes=CHECK,NO_PCH %s 
// RUN: %clang_cc1 -emit-llvm                                   -debug-info-macro %s -o - "-DC1(x)=( x  + 5 )" -DA -include %S/Inputs/debug-info-macro.h -UC1 | FileCheck -check-prefixes=NO_MACRO %s 

// RUN: %clang_cc1 -debug-info-kind=limited -debug-info-macro %S/Inputs/debug-info-macro.h -emit-pch -o %t.pch -DC3
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -debug-info-macro %s -o - -include-pch %t.pch "-DC1(x)=( x  + 5 )" -DA -include %S/Inputs/debug-info-macro.h -UC1 | FileCheck -check-prefixes=CHECK,PCH %s 

// This test checks that macro Debug info is correctly generated.

// TODO: Check for an following entry once support macros defined in pch files.
// -PCH: !DIMacro(type: DW_MACINFO_define, name: "C3", value: "1")>

#line 15
/*Line 15*/ #define D1 1
/*Line 16*/ #include  "Inputs/debug-info-macro.h"
/*Line 17*/ #undef D1
/*Line 18*/ #define D2 2
/*Line 19*/ #include  "Inputs/debug-info-macro.h"
/*Line 20*/ #undef D2

// NO_MACRO-NOT: macros
// NO_MACRO-NOT: DIMacro
// NO_MACRO-NOT: DIMacroFile

// CHECK:  !DICompileUnit({{.*}} macros: [[Macros:![0-9]+]]

// NO_PCH: [[Macros]] = !{[[MainMacroFile:![0-9]+]], [[BuiltinMacro:![0-9]+]], {{.*}}, [[DefineC1:![0-9]+]], [[DefineA:![0-9]+]], [[UndefC1:![0-9]+]]}
// PCH:    [[Macros]] = !{[[MainMacroFile:![0-9]+]], [[DefineC1:![0-9]+]], [[DefineA:![0-9]+]], [[UndefC1:![0-9]+]]}

// CHECK:  [[MainMacroFile]] = !DIMacroFile(file: [[MainFile:![0-9]+]], nodes: [[N1:![0-9]+]])
// CHECK:  [[MainFile]] = !DIFile(filename: "{{.*}}macro.c"
// CHECK:  [[N1]] = !{[[CommandLineInclude:![0-9]+]], [[DefineD1:![0-9]+]], [[FileInclude1:![0-9]+]], [[UndefD1:![0-9]+]], [[DefineD2:![0-9]+]], [[FileInclude2:![0-9]+]], [[UndefD2:![0-9]+]]}

// CHECK:  [[CommandLineInclude]] = !DIMacroFile(file: [[HeaderFile:![0-9]+]], nodes: [[N2:![0-9]+]])
// CHECK:  [[HeaderFile]] = !DIFile(filename: "{{.*}}debug-info-macro.h"
// CHECK:  [[N2]] = !{[[UndefA:![0-9]+]]}
// CHECK:  [[UndefA]] = !DIMacro(type: DW_MACINFO_undef, line: 11, name: "A")

// CHECK:  [[DefineD1]] = !DIMacro(type: DW_MACINFO_define, line: 15, name: "D1", value: "1")
// CHECK:  [[FileInclude1]] = !DIMacroFile(line: 16, file: [[HeaderFile]], nodes: [[N3:![0-9]+]])
// CHECK:  [[N3]] = !{[[DefineAx:![0-9]+]], [[UndefA]]}
// CHECK:  [[DefineAx]] = !DIMacro(type: DW_MACINFO_define, line: 3, name: "A(x,y,z)", value: "(x)")
// CHECK:  [[UndefD1]] = !DIMacro(type: DW_MACINFO_undef, line: 17, name: "D1") 

// CHECK:  [[DefineD2]] = !DIMacro(type: DW_MACINFO_define, line: 18, name: "D2", value: "2")
// CHECK:  [[FileInclude2]] = !DIMacroFile(line: 19, file: [[HeaderFile]], nodes: [[N4:![0-9]+]])
// CHECK:  [[N4]] = !{[[DefineAy:![0-9]+]], [[UndefA]]}
// CHECK:  [[DefineAy]] = !DIMacro(type: DW_MACINFO_define, line: 7, name: "A(x,y,z)", value: "(y)")
// CHECK:  [[UndefD2]] = !DIMacro(type: DW_MACINFO_undef, line: 20, name: "D2")

// NO_PCH: [[BuiltinMacro]] = !DIMacro(type: DW_MACINFO_define, name: "__llvm__", value: "1")

// CHECK:  [[DefineC1]] = !DIMacro(type: DW_MACINFO_define, name: "C1(x)", value: "( x + 5 )")
// CHECK:  [[DefineA]] = !DIMacro(type: DW_MACINFO_define, name: "A", value: "1")
// CHECK:  [[UndefC1]] = !DIMacro(type: DW_MACINFO_undef, name: "C1")
