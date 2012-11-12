// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -g | FileCheck %s

auto var = [](int i) { return i+1; };
void *use = &var;

extern "C" auto cvar = []{};

int a() { return []{ return 1; }(); }

int b(int x) { return [x]{return x;}(); }

int c(int x) { return [&x]{return x;}(); }

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; [x,y] { return y[x].x; }(); }

// Randomness for file. -- 6
// CHECK: [[FILE:.*]] = metadata !{i32 {{.*}}, metadata !{{.*}}debug-lambda-expressions.cpp{{.*}}; [ DW_TAG_file_type ]

// A: 10
// CHECK: [[A_FUNC:.*]] = metadata !{i32 {{.*}}, i32 0, metadata [[FILE]], metadata !"a", metadata !"a", metadata !"_Z1av", metadata {{.*}}, i32 [[A_LINE:.*]], metadata {{.*}}, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z1av, null, null, {{.*}} [ DW_TAG_subprogram ]

// B: 14
// CHECK: [[B_FUNC:.*]] = metadata !{i32 786478, i32 0, metadata [[FILE]], metadata !"b", metadata !"b", metadata !"_Z1bi", metadata [[FILE]], i32 [[B_LINE:.*]], metadata {{.*}}, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z1bi, null, null, {{.*}} ; [ DW_TAG_subprogram ]

// C: 17
// CHECK: [[C_FUNC:.*]] = metadata !{i32 {{.*}}, i32 0, metadata [[FILE]], metadata !"c", metadata !"c", metadata !"_Z1ci", metadata [[FILE]], i32 [[C_LINE:.*]], metadata {{.*}}, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z1ci, null, null, metadata {{.*}} ; [ DW_TAG_subprogram ]

// D: 18
// CHECK: [[D_FUNC:.*]] = metadata !{i32 {{.*}}, i32 0, metadata [[FILE]], metadata !"d", metadata !"d", metadata !"_Z1di", metadata [[FILE]], i32 [[D_LINE:.*]], metadata {{.*}}, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z1di, null, null, metadata {{.*}} ; [ DW_TAG_subprogram ]

// Back to D. -- 24
// CHECK: [[LAM_D:.*]] = metadata !{i32 {{.*}}, metadata [[D_FUNC]], metadata !"", metadata [[FILE]], i32 [[D_LINE]], i64 352, i64 32, i32 0, i32 0, null, metadata [[LAM_D_ARGS:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[LAM_D_ARGS]] = metadata !{metadata [[CAP_D_X:.*]], metadata [[CAP_D_Y:.*]], metadata [[CON_LAM_D:.*]], metadata [[DES_LAM_D:.*]]}
// CHECK: [[CAP_D_X]] = metadata !{i32 {{.*}}, metadata [[LAM_D]], metadata !"x", metadata [[FILE]], i32 [[D_LINE]], i64 32, i64 32, i64 0, i32 1, metadata {{.*}} ; [ DW_TAG_member ]
// CHECK: [[CAP_D_Y]] = metadata !{i32 {{.*}}, metadata [[LAM_D]], metadata !"y", metadata [[FILE]], i32 [[D_LINE]], i64 320, i64 32, i64 32, i32 1, metadata {{.*}} ; [ DW_TAG_member ]
// CHECK: [[CON_LAM_D]] = metadata {{.*}}[[LAM_D]], metadata !"operator()", metadata !"operator()"{{.*}}[ DW_TAG_subprogram ]
// CHECK: [[DES_LAM_D]] = metadata {{.*}}[[LAM_D]], metadata !"~", metadata !"~"{{.*}}[ DW_TAG_subprogram ]


// Back to C. -- 55
// CHECK: [[LAM_C:.*]] = metadata !{i32 {{.*}}, metadata [[C_FUNC]], metadata !"", metadata [[FILE]], i32 [[C_LINE]], i64 64, i64 64, i32 0, i32 0, null, metadata [[LAM_C_ARGS:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[LAM_C_ARGS]] = metadata !{metadata [[CAP_C:.*]], metadata [[CON_LAM_C:.*]], metadata [[DES_LAM_C:.*]]}
// Ignoring the member type for now.
// CHECK: [[CAP_C]] = metadata !{i32 {{.*}}, metadata [[LAM_C]], metadata !"x", metadata [[FILE]], i32 [[C_LINE]], i64 64, i64 64, i64 0, i32 1, metadata {{.*}}} ; [ DW_TAG_member ]
// CHECK: [[CON_LAM_C]] = metadata {{.*}}[[LAM_C]], metadata !"operator()", metadata !"operator()"{{.*}}[ DW_TAG_subprogram ]
// CHECK: [[DES_LAM_C]] = metadata {{.*}}[[LAM_C]], metadata !"~", metadata !"~"{{.*}}[ DW_TAG_subprogram ]


// Back to B. -- 67
// CHECK: [[LAM_B:.*]] = metadata !{i32 {{.*}}, metadata [[B_FUNC]], metadata !"", metadata [[FILE]], i32 [[B_LINE]], i64 32, i64 32, i32 0, i32 0, null, metadata [[LAM_B_ARGS:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[LAM_B_ARGS]] = metadata !{metadata [[CAP_B:.*]], metadata [[CON_LAM_B:.*]], metadata [[DES_LAM_B:.*]]}
// CHECK: [[CAP_B]] = metadata !{i32 {{.*}}, metadata [[LAM_B]], metadata !"x", metadata [[FILE]], i32 [[B_LINE]], i64 32, i64 32, i64 0, i32 1, metadata {{.*}}} ; [ DW_TAG_member ]
// CHECK: [[CON_LAM_B]] = metadata {{.*}}[[LAM_B]], metadata !"operator()", metadata !"operator()"{{.*}}[ DW_TAG_subprogram ]
// CHECK: [[DES_LAM_B]] = metadata {{.*}}[[LAM_B]], metadata !"~", metadata !"~"{{.*}}[ DW_TAG_subprogram ]

// Back to A. -- 78
// CHECK: [[LAM_A:.*]] = metadata !{i32 {{.*}}, metadata [[A_FUNC]], metadata !"", metadata [[FILE]], i32 [[A_LINE]], i64 8, i64 8, i32 0, i32 0, null, metadata [[LAM_A_ARGS:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[LAM_A_ARGS]] = metadata !{metadata [[CON_LAM_A:.*]], metadata [[DES_LAM_A:.*]]}
// CHECK: [[CON_LAM_A]] = metadata {{.*}}[[LAM_A]], metadata !"operator()", metadata !"operator()"{{.*}}[ DW_TAG_subprogram ]
// CHECK: [[DES_LAM_A]] = metadata {{.*}}[[LAM_A]], metadata !"~", metadata !"~"{{.*}}[ DW_TAG_subprogram ]

// CVAR:
// CHECK: metadata !{i32 {{.*}}, i32 0, null, metadata !"cvar", metadata !"cvar", metadata !"", metadata [[FILE]], i32 [[CVAR_LINE:.*]], metadata ![[CVAR_T:.*]], i32 0, i32 1, %class.anon.0* @cvar} ; [ DW_TAG_variable ]
// CHECK: [[CVAR_T]] = metadata !{i32 {{.*}}, null, metadata !"", metadata [[FILE]], i32 [[CVAR_LINE]], i64 8, i64 8, i32 0, i32 0, null, metadata ![[CVAR_ARGS:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[CVAR_ARGS]] = metadata !{metadata !{{.*}}, metadata !{{.*}}, metadata !{{.*}}}

// VAR:
// CHECK: metadata !{i32 {{.*}}, i32 0, null, metadata !"var", metadata !"var", metadata !"", metadata [[FILE]], i32 [[VAR_LINE:.*]], metadata ![[VAR_T:.*]], i32 1, i32 1, %class.anon* @var} ; [ DW_TAG_variable ]
// CHECK: [[VAR_T]] = metadata !{i32 {{.*}}, null, metadata !"", metadata [[FILE]], i32 [[VAR_LINE]], i64 8, i64 8, i32 0, i32 0, null, metadata ![[VAR_ARGS:.*]], i32 0, null, null} ; [ DW_TAG_class_type ]
// CHECK: [[VAR_ARGS]] = metadata !{metadata !{{.*}}, metadata !{{.*}}, metadata !{{.*}}}
