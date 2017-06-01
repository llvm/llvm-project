// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -debug-info-kind=limited | FileCheck %s

auto var = [](int i) { return i+1; };
void *use = &var;

extern "C" auto cvar = []{};

int a() { return []{ return 1; }(); }

int b(int x) { return [x]{return x;}(); }

int c(int x) { return [&x]{return x;}(); }

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; return [x,y] { return y[x].x; }(); }

// Randomness for file. -- 6

// VAR:
// CHECK: !DIGlobalVariable(name: "var"
// CHECK-SAME:              line: [[VAR_LINE:[0-9]+]]
// CHECK-SAME:              type: ![[VAR_T:[0-9]+]]

// CHECK: [[FILE:.*]] = !DIFile(filename: "{{.*}}debug-lambda-expressions.cpp",

// CVAR:
// CHECK: !DIGlobalVariable(name: "cvar"
// CHECK-SAME:              line: [[CVAR_LINE:[0-9]+]]
// CHECK-SAME:              type: ![[CVAR_T:[0-9]+]]
// CHECK: ![[CVAR_T]] = distinct !DICompositeType(tag: DW_TAG_class_type
// CHECK-SAME:                           line: [[CVAR_LINE]],
// CHECK-SAME:                           elements: ![[CVAR_ARGS:[0-9]+]]
// CHECK: ![[CVAR_ARGS]] = !{!{{[0-9]+}}}

// CHECK: ![[VAR_T]] = distinct !DICompositeType(tag: DW_TAG_class_type
// CHECK-SAME:                          line: [[VAR_LINE]],
// CHECK-SAME:                          elements: ![[VAR_ARGS:[0-9]+]]
// CHECK: ![[VAR_ARGS]] = !{!{{[0-9]+}}}

// CHECK: ![[INT:[0-9]+]] = !DIBasicType(name: "int"

// A: 10
// CHECK: ![[A_FUNC:.*]] = distinct !DISubprogram(name: "a"{{.*}}, line: [[A_LINE:[0-9]+]]{{.*}}, isDefinition: true

// Back to A. -- 78
// CHECK: ![[LAM_A:.*]] = distinct !DICompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[A_FUNC]]{{.*}}, line: [[A_LINE]],
// CHECK-SAME:                             elements: ![[LAM_A_ARGS:[0-9]+]]
// CHECK: ![[LAM_A_ARGS]] = !{![[CON_LAM_A:[0-9]+]]}
// CHECK: ![[CON_LAM_A]] = !DISubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_A]]
// CHECK-SAME:                           line: [[A_LINE]]
// CHECK-SAME:                           DIFlagPublic

// B: 14
// CHECK: ![[B_FUNC:.*]] = distinct !DISubprogram(name: "b"{{.*}}, line: [[B_LINE:[0-9]+]]{{.*}}, isDefinition: true

// Back to B. -- 67
// CHECK: ![[LAM_B:.*]] = distinct !DICompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[B_FUNC]]{{.*}}, line: [[B_LINE]],
// CHECK-SAME:                             elements: ![[LAM_B_ARGS:[0-9]+]]
// CHECK: ![[LAM_B_ARGS]] = !{![[CAP_B:[0-9]+]], ![[CON_LAM_B:[0-9]+]]}
// CHECK: ![[CAP_B]] = !DIDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:                        scope: ![[LAM_B]]
// CHECK-SAME:                        line: [[B_LINE]],
// CHECK-SAME:                        baseType: ![[INT]]
// CHECK: ![[CON_LAM_B]] = !DISubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_B]]
// CHECK-SAME:                           line: [[B_LINE]]
// CHECK-SAME:                           DIFlagPublic

// C: 17
// CHECK: ![[C_FUNC:.*]] = distinct !DISubprogram(name: "c"{{.*}}, line: [[C_LINE:[0-9]+]]{{.*}}, isDefinition: true

// Back to C. -- 55
// CHECK: ![[LAM_C:.*]] = distinct !DICompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[C_FUNC]]{{.*}}, line: [[C_LINE]],
// CHECK-SAME:                             elements: ![[LAM_C_ARGS:[0-9]+]]
// CHECK: ![[LAM_C_ARGS]] = !{![[CAP_C:[0-9]+]], ![[CON_LAM_C:[0-9]+]]}
// CHECK: ![[CAP_C]] = !DIDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:                        scope: ![[LAM_C]]
// CHECK-SAME:                        line: [[C_LINE]],
// CHECK-SAME:                        baseType: ![[TYPE_C_x:[0-9]+]]
// CHECK: ![[TYPE_C_x]] = !DIDerivedType(tag: DW_TAG_reference_type, baseType: ![[INT]]
// CHECK: ![[CON_LAM_C]] = !DISubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_C]]
// CHECK-SAME:                           line: [[C_LINE]]
// CHECK-SAME:                           DIFlagPublic

// D: 18
// CHECK: ![[D_FUNC:.*]] = distinct !DISubprogram(name: "d"{{.*}}, line: [[D_LINE:[0-9]+]]{{.*}}, isDefinition: true

// Back to D. -- 24
// CHECK: ![[LAM_D:.*]] = distinct !DICompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[D_FUNC]]{{.*}}, line: [[D_LINE]],
// CHECK-SAME:                             elements: ![[LAM_D_ARGS:[0-9]+]]
// CHECK: ![[LAM_D_ARGS]] = !{![[CAP_D_X:[0-9]+]], ![[CAP_D_Y:[0-9]+]], ![[CON_LAM_D:[0-9]+]]}
// CHECK: ![[CAP_D_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:                          scope: ![[LAM_D]]
// CHECK-SAME:                          line: [[D_LINE]],
// CHECK: ![[CAP_D_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y"
// CHECK-SAME:                          scope: ![[LAM_D]]
// CHECK-SAME:                          line: [[D_LINE]],
// CHECK: ![[CON_LAM_D]] = !DISubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_D]]
// CHECK-SAME:                           line: [[D_LINE]]
// CHECK-SAME:                           DIFlagPublic
