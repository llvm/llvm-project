// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// No TBAA without optimizations or with -fno-strict-aliasing.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-O0.cir
// RUN: FileCheck --input-file=%t-O0.cir %s -check-prefix=NO-TBAA
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -relaxed-aliasing -fclangir -emit-cir %s -o %t-relaxed.cir
// RUN: FileCheck --input-file=%t-relaxed.cir %s -check-prefix=NO-TBAA

// NO-TBAA-NOT: #cir.tbaa
// NO-TBAA-NOT: tbaa(

// CIR-DAG: #[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR-DAG: #[[INT:.*]] = #cir.tbaa_scalar<id = "int", size = 4>
// CIR-DAG: #[[LONG:.*]] = #cir.tbaa_scalar<id = "long", size = 8>
// CIR-DAG: #[[LLONG:.*]] = #cir.tbaa_scalar<id = "long long", size = 8>
// CIR-DAG: #[[FLOAT:.*]] = #cir.tbaa_scalar<id = "float", size = 4>
// CIR-DAG: #[[BOOL:.*]] = #cir.tbaa_scalar<id = "_Bool", size = 1>
// CIR-DAG: #[[BITINT:.*]] = #cir.tbaa_scalar<id = "_BitInt(17)", size = 4>
// CIR-DAG: #[[ANYPTR:.*]] = #cir.tbaa_scalar<id = "any pointer", size = 8>
// CIR-DAG: #[[ANYP2:.*]] = #cir.tbaa_scalar<id = "any p2 pointer", size = 8, parent = #[[ANYPTR]]>
// CIR-DAG: #[[P1INT:.*]] = #cir.tbaa_scalar<id = "p1 int", size = 8, parent = #[[ANYPTR]]>
// CIR-DAG: #[[P2INT:.*]] = #cir.tbaa_scalar<id = "p2 int", size = 8, parent = #[[ANYP2]]>
// CIR-DAG: #[[CHAR_TAG:.*]] = #cir.tbaa_tag<base = #[[CHAR]], access = #[[CHAR]], offset = 0>
// CIR-DAG: #[[INT_TAG:.*]] = #cir.tbaa_tag<base = #[[INT]], access = #[[INT]], offset = 0>
// CIR-DAG: #[[LONG_TAG:.*]] = #cir.tbaa_tag<base = #[[LONG]], access = #[[LONG]], offset = 0>
// CIR-DAG: #[[LLONG_TAG:.*]] = #cir.tbaa_tag<base = #[[LLONG]], access = #[[LLONG]], offset = 0>
// CIR-DAG: #[[FLOAT_TAG:.*]] = #cir.tbaa_tag<base = #[[FLOAT]], access = #[[FLOAT]], offset = 0>
// CIR-DAG: #[[BOOL_TAG:.*]] = #cir.tbaa_tag<base = #[[BOOL]], access = #[[BOOL]], offset = 0>
// CIR-DAG: #[[BITINT_TAG:.*]] = #cir.tbaa_tag<base = #[[BITINT]], access = #[[BITINT]], offset = 0>
// CIR-DAG: #[[ANYPTR_TAG:.*]] = #cir.tbaa_tag<base = #[[ANYPTR]], access = #[[ANYPTR]], offset = 0>
// CIR-DAG: #[[P1INT_TAG:.*]] = #cir.tbaa_tag<base = #[[P1INT]], access = #[[P1INT]], offset = 0>
// CIR-DAG: #[[P2INT_TAG:.*]] = #cir.tbaa_tag<base = #[[P2INT]], access = #[[P2INT]], offset = 0>

// Signed and unsigned types share a type descriptor; long and long long do
// not, although both are 64 bits wide.
void integers(int *i, unsigned *u, long *l, long long *ll) {
  *i = 1;
  *u = 2;
  *l = 3;
  *ll = 4;
}

// CIR-LABEL: cir.func {{.*}}@integers
// CIR: cir.store align(4) %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#[[INT_TAG]])
// CIR: cir.store align(4) %{{.*}}, %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#[[INT_TAG]])
// CIR: cir.store align(8) %{{.*}}, %{{.*}} : !s64i, !cir.ptr<!s64i> tbaa(#[[LONG_TAG]])
// CIR: cir.store align(8) %{{.*}}, %{{.*}} : !s64i, !cir.ptr<!s64i> tbaa(#[[LLONG_TAG]])

// LLVM-LABEL: define {{.*}}void @integers(
// LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[INT_TAG:![0-9]+]]
// LLVM: store i32 2, ptr %{{.*}}, align 4, !tbaa [[INT_TAG]]
// LLVM: store i64 3, ptr %{{.*}}, align 8, !tbaa [[LONG_TAG:![0-9]+]]
// LLVM: store i64 4, ptr %{{.*}}, align 8, !tbaa [[LLONG_TAG:![0-9]+]]

typedef int __attribute__((may_alias)) aliasing_int;

// Character types and may_alias types are in the omnipotent char class.
int chars(char *c, unsigned char *uc, aliasing_int *ai) {
  *c = 1;
  *uc = 2;
  return *ai;
}

// CIR-LABEL: cir.func {{.*}}@chars
// CIR: cir.store align(1) %{{.*}}, %{{.*}} : !s8i, !cir.ptr<!s8i> tbaa(#[[CHAR_TAG]])
// CIR: cir.store align(1) %{{.*}}, %{{.*}} : !u8i, !cir.ptr<!u8i> tbaa(#[[CHAR_TAG]])
// CIR: cir.load align(4) %{{.*}} : !cir.ptr<!s32i>, !s32i tbaa(#[[CHAR_TAG]])

// LLVM-LABEL: define {{.*}}i32 @chars(
// LLVM: store i8 1, ptr %{{.*}}, align 1, !tbaa [[CHAR_TAG:![0-9]+]]
// LLVM: store i8 2, ptr %{{.*}}, align 1, !tbaa [[CHAR_TAG]]
// LLVM: load i32, ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG]]

enum E { A, B };

// In C, enums use the type descriptor of their underlying type.
void others(float *f, _Bool *b, _BitInt(17) *bi, enum E *e) {
  *f = 1.0f;
  *b = 1;
  *bi = 3;
  *e = B;
}

// CIR-LABEL: cir.func {{.*}}@others
// CIR: cir.store align(4) %{{.*}}, %{{.*}} : !cir.float, !cir.ptr<!cir.float> tbaa(#[[FLOAT_TAG]])
// CIR: cir.store align(1) %{{.*}}, %{{.*}} : !cir.bool, !cir.ptr<!cir.bool> tbaa(#[[BOOL_TAG]])
// CIR: cir.store align(4) %{{.*}}, %{{.*}} : !cir.int<s, 17, bitint>, !cir.ptr<!cir.int<s, 17, bitint>> tbaa(#[[BITINT_TAG]])
// CIR: cir.store align(4) %{{.*}}, %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#[[INT_TAG]])

// LLVM-LABEL: define {{.*}}void @others(
// LLVM: store float 1.000000e+00, ptr %{{.*}}, align 4, !tbaa [[FLOAT_TAG:![0-9]+]]
// LLVM: store i8 1, ptr %{{.*}}, align 1, !tbaa [[BOOL_TAG:![0-9]+]]
// LLVM: store i32 3, ptr %{{.*}}, align 4, !tbaa [[BITINT_TAG:![0-9]+]]
// LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[INT_TAG]]

// Pointers are distinguished by their pointer depth and base pointee type;
// void pointers use the "any pointer" type descriptors.
void pointers(int **pp, void **vp, int ***ppp) {
  *pp = 0;
  *vp = 0;
  **ppp = 0;
}

// CIR-LABEL: cir.func {{.*}}@pointers
// CIR: cir.store {{.*}} : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>> tbaa(#[[P1INT_TAG]])
// CIR: cir.store {{.*}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>> tbaa(#[[ANYPTR_TAG]])
// CIR: cir.load {{.*}} : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>> tbaa(#[[P2INT_TAG]])
// CIR: cir.store {{.*}} : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>> tbaa(#[[P1INT_TAG]])

// LLVM-LABEL: define {{.*}}void @pointers(
// LLVM: store ptr null, ptr %{{.*}}, align 8, !tbaa [[P1INT_TAG:![0-9]+]]
// LLVM: store ptr null, ptr %{{.*}}, align 8, !tbaa [[ANYPTR_TAG:![0-9]+]]
// LLVM-NEXT: load ptr, ptr %{{.*}}, align 8, !tbaa [[P3INT_TAG:![0-9]+]]
// LLVM-NEXT: load ptr, ptr %{{.*}}, align 8, !tbaa [[P2INT_TAG:![0-9]+]]
// LLVM-NEXT: store ptr null, ptr %{{.*}}, align 8, !tbaa [[P1INT_TAG]]

// Array subscripts access objects of the element type.
long arrays(long *p, int i) {
  long a[4];
  a[i] = p[i];
  return a[0];
}

// CIR-LABEL: cir.func {{.*}}@arrays
// CIR: cir.load {{.*}} : !cir.ptr<!s64i>, !s64i tbaa(#[[LONG_TAG]])
// CIR: cir.store {{.*}} : !s64i, !cir.ptr<!s64i> tbaa(#[[LONG_TAG]])
// CIR: cir.load {{.*}} : !cir.ptr<!s64i>, !s64i tbaa(#[[LONG_TAG]])

// LLVM-LABEL: define {{.*}}i64 @arrays(
// LLVM: load i64, ptr %{{.*}}, align 8, !tbaa [[LONG_TAG]]
// LLVM: store i64 %{{.*}}, ptr %{{.*}}, align 8, !tbaa [[LONG_TAG]]
// LLVM: load i64, ptr %{{.*}}, align 16, !tbaa [[LONG_TAG]]

// __builtin_bit_cast reads the source object through an omnipotent char
// access.
float bitcast(int x) { return __builtin_bit_cast(float, x); }

// CIR-LABEL: cir.func {{.*}}@bitcast
// CIR: cir.load {{.*}} : !cir.ptr<!cir.float>, !cir.float tbaa(#[[CHAR_TAG]])

// LLVM-LABEL: define {{.*}}float @bitcast(
// LLVM: load float, ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG]]

// LLVM-DAG: [[ROOT:![0-9]+]] = !{!"Simple C/C++ TBAA"}
// LLVM-DAG: [[CHAR:![0-9]+]] = !{!"omnipotent char", [[ROOT]], i64 0}
// LLVM-DAG: [[INT:![0-9]+]] = !{!"int", [[CHAR]], i64 0}
// LLVM-DAG: [[LONG:![0-9]+]] = !{!"long", [[CHAR]], i64 0}
// LLVM-DAG: [[LLONG:![0-9]+]] = !{!"long long", [[CHAR]], i64 0}
// LLVM-DAG: [[FLOAT:![0-9]+]] = !{!"float", [[CHAR]], i64 0}
// LLVM-DAG: [[BOOL:![0-9]+]] = !{!"_Bool", [[CHAR]], i64 0}
// LLVM-DAG: [[BITINT:![0-9]+]] = !{!"_BitInt(17)", [[CHAR]], i64 0}
// LLVM-DAG: [[ANYPTR:![0-9]+]] = !{!"any pointer", [[CHAR]], i64 0}
// LLVM-DAG: [[ANYP2:![0-9]+]] = !{!"any p2 pointer", [[ANYPTR]], i64 0}
// LLVM-DAG: [[P1INT:![0-9]+]] = !{!"p1 int", [[ANYPTR]], i64 0}
// LLVM-DAG: [[P2INT:![0-9]+]] = !{!"p2 int", [[ANYP2]], i64 0}
// LLVM-DAG: [[ANYP3:![0-9]+]] = !{!"any p3 pointer", [[ANYP2]], i64 0}
// LLVM-DAG: [[P3INT:![0-9]+]] = !{!"p3 int", [[ANYP3]], i64 0}
// LLVM-DAG: [[CHAR_TAG]] = !{[[CHAR]], [[CHAR]], i64 0}
// LLVM-DAG: [[INT_TAG]] = !{[[INT]], [[INT]], i64 0}
// LLVM-DAG: [[LONG_TAG]] = !{[[LONG]], [[LONG]], i64 0}
// LLVM-DAG: [[LLONG_TAG]] = !{[[LLONG]], [[LLONG]], i64 0}
// LLVM-DAG: [[FLOAT_TAG]] = !{[[FLOAT]], [[FLOAT]], i64 0}
// LLVM-DAG: [[BOOL_TAG]] = !{[[BOOL]], [[BOOL]], i64 0}
// LLVM-DAG: [[BITINT_TAG]] = !{[[BITINT]], [[BITINT]], i64 0}
// LLVM-DAG: [[ANYPTR_TAG]] = !{[[ANYPTR]], [[ANYPTR]], i64 0}
// LLVM-DAG: [[P1INT_TAG]] = !{[[P1INT]], [[P1INT]], i64 0}
// LLVM-DAG: [[P2INT_TAG]] = !{[[P2INT]], [[P2INT]], i64 0}
// LLVM-DAG: [[P3INT_TAG]] = !{[[P3INT]], [[P3INT]], i64 0}
