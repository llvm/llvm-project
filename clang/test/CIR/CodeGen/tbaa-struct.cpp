// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Without struct-path TBAA, member accesses use scalar access tags.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -no-struct-path-tbaa -fclangir -emit-cir %s -o %t-no-path.cir
// RUN: FileCheck --input-file=%t-no-path.cir %s -check-prefix=NO-PATH

struct Inner { short x; float y; };
struct S { int a; long b; Inner in; int arr[4]; };
struct Base { int b; };
struct Derived : Base { long d; };
union U { int i; float f; };
struct WithRef { int &r; };
enum class Color : short { Red };

// CIR-DAG: #[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR-DAG: #[[INT:.*]] = #cir.tbaa_scalar<id = "int", size = 4>
// CIR-DAG: #[[LONG:.*]] = #cir.tbaa_scalar<id = "long", size = 8>
// CIR-DAG: #[[SHORT:.*]] = #cir.tbaa_scalar<id = "short", size = 2>
// CIR-DAG: #[[FLOAT:.*]] = #cir.tbaa_scalar<id = "float", size = 4>
// CIR-DAG: #[[COLOR:.*]] = #cir.tbaa_scalar<id = "_ZTS5Color", size = 2>
// CIR-DAG: #[[ANYPTR:.*]] = #cir.tbaa_scalar<id = "any pointer", size = 8>
// CIR-DAG: #[[P1INT:.*]] = #cir.tbaa_scalar<id = "p1 int", size = 8, parent = #[[ANYPTR]]>
// CIR-DAG: #[[INNER:.*]] = #cir.tbaa_struct<"_ZTS5Inner", [<#[[SHORT]], 0>, <#[[FLOAT]], 4>]>
// CIR-DAG: #[[S:.*]] = #cir.tbaa_struct<"_ZTS1S", [<#[[INT]], 0>, <#[[LONG]], 8>, <#[[INNER]], 16>, <#[[CHAR]], 24>]>
// CIR-DAG: #[[BASE:.*]] = #cir.tbaa_struct<"_ZTS4Base", [<#[[INT]], 0>]>
// CIR-DAG: #[[DERIVED:.*]] = #cir.tbaa_struct<"_ZTS7Derived", [<#[[BASE]], 0>, <#[[LONG]], 8>]>
// CIR-DAG: #[[WITHREF:.*]] = #cir.tbaa_struct<"_ZTS7WithRef", [<#[[P1INT]], 0>]>
// CIR-DAG: #[[CHAR_TAG:.*]] = #cir.tbaa_tag<base = #[[CHAR]], access = #[[CHAR]], offset = 0>
// CIR-DAG: #[[INT_TAG:.*]] = #cir.tbaa_tag<base = #[[INT]], access = #[[INT]], offset = 0>
// CIR-DAG: #[[COLOR_TAG:.*]] = #cir.tbaa_tag<base = #[[COLOR]], access = #[[COLOR]], offset = 0>
// CIR-DAG: #[[S_A_TAG:.*]] = #cir.tbaa_tag<base = #[[S]], access = #[[INT]], offset = 0>
// CIR-DAG: #[[S_B_TAG:.*]] = #cir.tbaa_tag<base = #[[S]], access = #[[LONG]], offset = 8>
// CIR-DAG: #[[S_IN_X_TAG:.*]] = #cir.tbaa_tag<base = #[[S]], access = #[[SHORT]], offset = 16>
// CIR-DAG: #[[S_IN_Y_TAG:.*]] = #cir.tbaa_tag<base = #[[S]], access = #[[FLOAT]], offset = 20>
// CIR-DAG: #[[BASE_B_TAG:.*]] = #cir.tbaa_tag<base = #[[BASE]], access = #[[INT]], offset = 0>
// CIR-DAG: #[[DERIVED_D_TAG:.*]] = #cir.tbaa_tag<base = #[[DERIVED]], access = #[[LONG]], offset = 8>
// CIR-DAG: #[[WITHREF_R_TAG:.*]] = #cir.tbaa_tag<base = #[[WITHREF]], access = #[[P1INT]], offset = 0>

// NO-PATH-DAG: #[[NP_INT:.*]] = #cir.tbaa_scalar<id = "int", size = 4>
// NO-PATH-DAG: #[[NP_LONG:.*]] = #cir.tbaa_scalar<id = "long", size = 8>
// NO-PATH-DAG: #[[NP_INT_TAG:.*]] = #cir.tbaa_tag<base = #[[NP_INT]], access = #[[NP_INT]], offset = 0>
// NO-PATH-DAG: #[[NP_LONG_TAG:.*]] = #cir.tbaa_tag<base = #[[NP_LONG]], access = #[[NP_LONG]], offset = 0>

void members(S *s) {
  s->a = 1;
  s->b = 2;
}

// CIR-LABEL: cir.func {{.*}}@_Z7membersP1S
// CIR: cir.store {{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#[[S_A_TAG]])
// CIR: cir.store {{.*}} : !s64i, !cir.ptr<!s64i> tbaa(#[[S_B_TAG]])

// LLVM-LABEL: define {{.*}}void @_Z7membersP1S(
// LLVM: store i32 1, ptr %{{.*}}, align 8, !tbaa [[S_A_TAG:![0-9]+]]
// LLVM: store i64 2, ptr %{{.*}}, align 8, !tbaa [[S_B_TAG:![0-9]+]]

// NO-PATH-LABEL: cir.func {{.*}}@_Z7membersP1S
// NO-PATH: cir.store {{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#[[NP_INT_TAG]])
// NO-PATH: cir.store {{.*}} : !s64i, !cir.ptr<!s64i> tbaa(#[[NP_LONG_TAG]])

// Offsets of nested members are relative to the outermost struct.
float nested(S &s) {
  s.in.x = 1;
  return s.in.y;
}

// CIR-LABEL: cir.func {{.*}}@_Z6nestedR1S
// CIR: cir.store {{.*}} : !s16i, !cir.ptr<!s16i> tbaa(#[[S_IN_X_TAG]])
// CIR: cir.load {{.*}} : !cir.ptr<!cir.float>, !cir.float tbaa(#[[S_IN_Y_TAG]])

// LLVM-LABEL: define {{.*}}float @_Z6nestedR1S(
// LLVM: store i16 1, ptr %{{.*}}, align 8, !tbaa [[S_IN_X_TAG:![0-9]+]]
// LLVM: load float, ptr %{{.*}}, align 4, !tbaa [[S_IN_Y_TAG:![0-9]+]]

// Accesses to elements of member arrays are not struct-path accesses.
int memberArray(S *s, int i) { return s->arr[i]; }

// CIR-LABEL: cir.func {{.*}}@_Z11memberArrayP1Si
// CIR: cir.get_element
// CIR-NEXT: cir.load {{.*}} : !cir.ptr<!s32i>, !s32i tbaa(#[[INT_TAG]])

// LLVM-LABEL: define {{.*}}i32 @_Z11memberArrayP1Si(
// LLVM: getelementptr
// LLVM: load i32, ptr %{{.*}}, align 4, !tbaa [[INT_TAG:![0-9]+]]

long derived(Derived *d) {
  d->b = 1;
  return d->d;
}

// CIR-LABEL: cir.func {{.*}}@_Z7derivedP7Derived
// CIR: cir.store {{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#[[BASE_B_TAG]])
// CIR: cir.load {{.*}} : !cir.ptr<!s64i>, !s64i tbaa(#[[DERIVED_D_TAG]])

// LLVM-LABEL: define {{.*}}i64 @_Z7derivedP7Derived(
// LLVM: store i32 1, ptr %{{.*}}, align 8, !tbaa [[BASE_B_TAG:![0-9]+]]
// LLVM: load i64, ptr %{{.*}}, align 8, !tbaa [[DERIVED_D_TAG:![0-9]+]]

// Union members are accessed as omnipotent char.
float unions(U *u) {
  u->i = 1;
  return u->f;
}

// CIR-LABEL: cir.func {{.*}}@_Z6unionsP1U
// CIR: cir.store {{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#[[CHAR_TAG]])
// CIR: cir.load {{.*}} : !cir.ptr<!cir.float>, !cir.float tbaa(#[[CHAR_TAG]])

// LLVM-LABEL: define {{.*}}float @_Z6unionsP1U(
// LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG:![0-9]+]]
// LLVM: load float, ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG]]

// The reference member is loaded as a pointer member of the struct; the
// referenced object is accessed through its own type.
void references(WithRef &w) { w.r = 1; }

// CIR-LABEL: cir.func {{.*}}@_Z10referencesR7WithRef
// CIR: cir.load {{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i> tbaa(#[[WITHREF_R_TAG]])
// CIR: cir.store {{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#[[INT_TAG]])

// LLVM-LABEL: define {{.*}}void @_Z10referencesR7WithRef(
// LLVM: getelementptr inbounds nuw %struct.WithRef
// LLVM-NEXT: load ptr, ptr %{{.*}}, align 8, !tbaa [[WITHREF_R_TAG:![0-9]+]]
// LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[INT_TAG]]

// C++ enums with external linkage are distinct types; others are omnipotent
// char.
void enums(Color *c) {
  enum Local { L } l;
  *c = Color::Red;
  l = L;
}

// CIR-LABEL: cir.func {{.*}}@_Z5enumsP5Color
// CIR: cir.store {{.*}} : !s16i, !cir.ptr<!s16i> tbaa(#[[COLOR_TAG]])
// CIR: cir.store {{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#[[CHAR_TAG]])

// LLVM-LABEL: define {{.*}}void @_Z5enumsP5Color(
// LLVM: store i16 0, ptr %{{.*}}, align 2, !tbaa [[COLOR_TAG:![0-9]+]]
// LLVM: store i32 0, ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG]]

// LLVM-DAG: [[ROOT:![0-9]+]] = !{!"Simple C++ TBAA"}
// LLVM-DAG: [[CHAR:![0-9]+]] = !{!"omnipotent char", [[ROOT]], i64 0}
// LLVM-DAG: [[INT:![0-9]+]] = !{!"int", [[CHAR]], i64 0}
// LLVM-DAG: [[LONG:![0-9]+]] = !{!"long", [[CHAR]], i64 0}
// LLVM-DAG: [[SHORT:![0-9]+]] = !{!"short", [[CHAR]], i64 0}
// LLVM-DAG: [[FLOAT:![0-9]+]] = !{!"float", [[CHAR]], i64 0}
// LLVM-DAG: [[COLOR:![0-9]+]] = !{!"_ZTS5Color", [[CHAR]], i64 0}
// LLVM-DAG: [[ANYPTR:![0-9]+]] = !{!"any pointer", [[CHAR]], i64 0}
// LLVM-DAG: [[P1INT:![0-9]+]] = !{!"p1 int", [[ANYPTR]], i64 0}
// LLVM-DAG: [[INNER:![0-9]+]] = !{!"_ZTS5Inner", [[SHORT]], i64 0, [[FLOAT]], i64 4}
// LLVM-DAG: [[S:![0-9]+]] = !{!"_ZTS1S", [[INT]], i64 0, [[LONG]], i64 8, [[INNER]], i64 16, [[CHAR]], i64 24}
// LLVM-DAG: [[BASE:![0-9]+]] = !{!"_ZTS4Base", [[INT]], i64 0}
// LLVM-DAG: [[DERIVED:![0-9]+]] = !{!"_ZTS7Derived", [[BASE]], i64 0, [[LONG]], i64 8}
// LLVM-DAG: [[WITHREF:![0-9]+]] = !{!"_ZTS7WithRef", [[P1INT]], i64 0}
// LLVM-DAG: [[CHAR_TAG]] = !{[[CHAR]], [[CHAR]], i64 0}
// LLVM-DAG: [[INT_TAG]] = !{[[INT]], [[INT]], i64 0}
// LLVM-DAG: [[COLOR_TAG]] = !{[[COLOR]], [[COLOR]], i64 0}
// LLVM-DAG: [[S_A_TAG]] = !{[[S]], [[INT]], i64 0}
// LLVM-DAG: [[S_B_TAG]] = !{[[S]], [[LONG]], i64 8}
// LLVM-DAG: [[S_IN_X_TAG]] = !{[[S]], [[SHORT]], i64 16}
// LLVM-DAG: [[S_IN_Y_TAG]] = !{[[S]], [[FLOAT]], i64 20}
// LLVM-DAG: [[BASE_B_TAG]] = !{[[BASE]], [[INT]], i64 0}
// LLVM-DAG: [[DERIVED_D_TAG]] = !{[[DERIVED]], [[LONG]], i64 8}
// LLVM-DAG: [[WITHREF_R_TAG]] = !{[[WITHREF]], [[P1INT]], i64 0}
