// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - -O1 %s | \
// RUN:     FileCheck -check-prefixes=CHECK,CHECK-OLD %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -new-struct-path-tbaa \
// RUN:     -emit-llvm -o - -O1 %s | \
// RUN:     FileCheck -check-prefixes=CHECK,CHECK-NEW %s
//
// Check that we generate TBAA metadata for struct copies correctly.

struct A {
  short s;
  int i;
  char c;
  int j;
};

typedef A __attribute__((may_alias)) AA;

void copy(A *a1, A *a2) {
// CHECK-LABEL: _Z4copyP1AS0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %{{.*}}, ptr noundef nonnull align 4 dereferenceable(16) %{{.*}}, i64 16, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_A:![0-9]*]]
  *a1 = *a2;
}

struct B {
  char c;
  A a;
  int i;
};

void copy2(B *b1, B *b2) {
// CHECK-LABEL: _Z5copy2P1BS0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(24) %{{.*}}, ptr noundef nonnull align 4 dereferenceable(24) %{{.*}}, i64 24, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS2:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_B:![0-9]*]]
  *b1 = *b2;
}

struct S {
  _Complex char cc;
  _Complex int ci;
};

union U {
  _Complex int ci;
  S s;
};

void copy3(U *u1, U *u2) {
// CHECK-LABEL: _Z5copy3P1US0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(12) %{{.*}}, ptr noundef nonnull align 4 dereferenceable(12) %{{.*}}, i64 12, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS3:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_U:![0-9]*]]
  *u1 = *u2;
}

// Make sure that zero-length bitfield works.
struct C {
  char a;
  int : 0;  // Shall not be ignored; see r185018.
  char b;
  char c;
} __attribute__((ms_struct));

void copy4(C *c1, C *c2) {
// CHECK-LABEL: _Z5copy4P1CS0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) {{.*}}, ptr noundef nonnull align 1 dereferenceable(3) {{.*}}, i64 3, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS4:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_C:![0-9]*]]
  *c1 = *c2;
}

struct D {
  char a;
  int : 0;
  char b;
  char c;
};

void copy5(D *d1, D *d2) {
// CHECK-LABEL: _Z5copy5P1DS0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) {{.*}}, ptr noundef nonnull align 1 dereferenceable(6) {{.*}}, i64 6, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS5:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_D:![0-9]*]]
  *d1 = *d2;
}

void copy6(AA *a1, A *a2) {
// CHECK-LABEL: _Z5copy6P1AS0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %{{.*}}, ptr noundef nonnull align 4 dereferenceable(16) %{{.*}}, i64 16, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS]]
// CHECK-NEW-SAME: !tbaa [[TAG_char:![0-9]*]]
  *a1 = *a2;
}

void copy7(A *a1, AA *a2) {
// CHECK-LABEL: _Z5copy7P1AS0_
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %{{.*}}, ptr noundef nonnull align 4 dereferenceable(16) %{{.*}}, i64 16, i1 false)
// CHECK-OLD-SAME: !tbaa.struct [[TS]]
// CHECK-NEW-SAME: !tbaa [[TAG_char]]
  *a1 = *a2;
}

struct NamedBitfields {
   signed f0 : 9;
   unsigned f1 : 2;
   char f2;
   double f3;
};

void copy8(NamedBitfields *a1, NamedBitfields *a2) {
// CHECK-LABEL: _Z5copy8P14NamedBitfieldsS0_
// CHECK:   tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %a1, ptr noundef nonnull align 8 dereferenceable(16) %a2, i64 16, i1 false),
// CHECK-OLD-SAME: !tbaa.struct [[TS6:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_NamedBitfields:!.+]], !tbaa.struct
  *a1 = *a2;
}

struct NamedBitfields2 {
  char a, b, c;
   signed f0 : 3;
   unsigned f1 : 4;
   char f2 : 7;
   double f3;
   unsigned f4 : 4;
};

void copy9(NamedBitfields2 *a1, NamedBitfields2 *a2) {
// CHECK-LABEL: _Z5copy9P15NamedBitfields2S0_
// CHECK:   tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %a1, ptr noundef nonnull align 8 dereferenceable(24) %a2, i64 24, i1 false),
// CHECK-OLD-SAME: !tbaa.struct [[TS7:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_NamedBitfields2:!.+]], !tbaa.struct
  *a1 = *a2;
}

// Test with unnamed bitfield at the start and in between named ones..
struct NamedBitfields3 {
  unsigned : 11;
  signed f0 : 9;
  char : 2;
  int f1 : 2;
  double f2;
};

void copy10(NamedBitfields3 *a1, NamedBitfields3 *a2) {
// CHECK-LABEL: _Z6copy10P15NamedBitfields3S0_
// CHECK:   tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %a1, ptr noundef nonnull align 8 dereferenceable(16) %a2, i64 16, i1 false),
// CHECK-OLD-SAME: !tbaa.struct [[TS8:!.*]]
// CHECK-NEW-SAME: !tbaa [[TAG_NamedBitfields3:!.+]], !tbaa.struct
  *a1 = *a2;
}

// CHECK-OLD: [[TS]] = !{i64 0, i64 2, !{{.*}}, i64 4, i64 4, !{{.*}}, i64 8, i64 1, !{{.*}}, i64 12, i64 4, !{{.*}}}
// CHECK-OLD: [[CHAR:!.*]] = !{!"omnipotent char", !{{.*}}}
// CHECK-OLD: [[TAG_INT:!.*]] = !{[[INT:!.*]], [[INT]], i64 0}
// CHECK-OLD: [[INT]] = !{!"int", [[CHAR]]
// CHECK-OLD: [[TAG_CHAR:!.*]] = !{[[CHAR]], [[CHAR]], i64 0}
// (offset, size) = (0,1) char; (4,2) short; (8,4) int; (12,1) char; (16,4) int; (20,4) int
// CHECK-OLD: [[TS2]] = !{i64 0, i64 1, !{{.*}}, i64 4, i64 2, !{{.*}}, i64 8, i64 4, !{{.*}}, i64 12, i64 1, !{{.*}}, i64 16, i64 4, {{.*}}, i64 20, i64 4, {{.*}}}
// (offset, size) = (0,8) char; (0,2) char; (4,8) char
// CHECK-OLD: [[TS3]] = !{i64 0, i64 8, !{{.*}}, i64 0, i64 2, !{{.*}}, i64 4, i64 8, !{{.*}}}
// CHECK-OLD: [[TS4]] = !{i64 0, i64 1, [[TAG_CHAR]], i64 1, i64 1, [[TAG_CHAR]], i64 2, i64 1, [[TAG_CHAR]]}
// CHECK-OLD: [[TS5]] = !{i64 0, i64 1, [[TAG_CHAR]], i64 4, i64 1, [[TAG_CHAR]], i64 5, i64 1, [[TAG_CHAR]]}
// CHECK-OLD: [[TS6]] = !{i64 0, i64 2, [[TAG_CHAR]], i64 2, i64 1, [[TAG_CHAR]], i64 8, i64 8, [[TAG_DOUBLE:!.+]]}
// CHECK-OLD: [[TAG_DOUBLE]] = !{[[DOUBLE:!.+]], [[DOUBLE]], i64 0}
// CHECK-OLD  [[DOUBLE]] = !{!"double", [[CHAR]], i64 0}
// CHECK-OLD: [[TS7]] = !{i64 0, i64 1, [[TAG_CHAR]], i64 1, i64 1, [[TAG_CHAR]], i64 2, i64 1, [[TAG_CHAR]], i64 3, i64 1, [[TAG_CHAR]], i64 4, i64 1, [[TAG_CHAR]], i64 8, i64 8, [[TAG_DOUBLE]], i64 16, i64 1, [[TAG_CHAR]]}
// CHECK-OLD: [[TS8]] = !{i64 0, i64 4, [[TAG_CHAR]], i64 8, i64 8, [[TAG_DOUBLE]]}

// CHECK-NEW-DAG: [[TYPE_char:!.*]] = !{{{.*}}, i64 1, !"omnipotent char"}
// CHECK-NEW-DAG: [[TAG_char]] = !{[[TYPE_char]], [[TYPE_char]], i64 0, i64 0}
// CHECK-NEW-DAG: [[TYPE_short:!.*]] = !{[[TYPE_char]], i64 2, !"short"}
// CHECK-NEW-DAG: [[TYPE_int:!.*]] = !{[[TYPE_char]], i64 4, !"int"}
// CHECK-NEW-DAG: [[TYPE_A:!.*]] = !{[[TYPE_char]], i64 16, !"_ZTS1A", [[TYPE_short]], i64 0, i64 2, [[TYPE_int]], i64 4, i64 4, [[TYPE_char]], i64 8, i64 1, [[TYPE_int]], i64 12, i64 4}
// CHECK-NEW-DAG: [[TAG_A]] = !{[[TYPE_A]], [[TYPE_A]], i64 0, i64 16}
// CHECK-NEW-DAG: [[TYPE_B:!.*]] = !{[[TYPE_char]], i64 24, !"_ZTS1B", [[TYPE_char]], i64 0, i64 1, [[TYPE_A]], i64 4, i64 16, [[TYPE_int]], i64 20, i64 4}
// CHECK-NEW-DAG: [[TAG_B]] = !{[[TYPE_B]], [[TYPE_B]], i64 0, i64 24}
// CHECK-NEW-DAG: [[TAG_U]] = !{[[TYPE_char]], [[TYPE_char]], i64 0, i64 12}
// CHECK-NEW-DAG: [[TYPE_C:!.*]] = !{[[TYPE_char]], i64 3, !"_ZTS1C", [[TYPE_char]], i64 0, i64 1, [[TYPE_char]], i64 1, i64 1, [[TYPE_char]], i64 2, i64 1}
// CHECK-NEW-DAG: [[TAG_C]] = !{[[TYPE_C]], [[TYPE_C]], i64 0, i64 3}
// CHECK-NEW-DAG: [[TYPE_D:!.*]] = !{[[TYPE_char]], i64 6, !"_ZTS1D", [[TYPE_char]], i64 0, i64 1, [[TYPE_char]], i64 4, i64 1, [[TYPE_char]], i64 5, i64 1}
// CHECK-NEW-DAG: [[TAG_D]] = !{[[TYPE_D]], [[TYPE_D]], i64 0, i64 6}
// CHECK-NEW-DAG: [[TAG_NamedBitfields]] = !{[[TYPE_NamedBitfields:!.+]], [[TYPE_NamedBitfields]], i64 0, i64 16}
// CHECK-NEW-DAG: [[TYPE_NamedBitfields]] = !{[[TYPE_char]], i64 16, !"_ZTS14NamedBitfields", [[TYPE_int]], i64 0, i64 4, [[TYPE_int]], i64 1, i64 4, [[TYPE_char]], i64 2, i64 1, [[TYPE_double:!.+]], i64 8, i64 8}
// CHECK-NEW-DAG: [[TYPE_double]] = !{[[TYPE_char]], i64 8, !"double"}
// CHECK-NEW-DAG: [[TAG_NamedBitfields2]] = !{[[TYPE_NamedBitfields2:!.+]], [[TYPE_NamedBitfields2]], i64 0, i64 24}
// CHECK-NEW-DAG: [[TYPE_NamedBitfields2]] = !{[[TYPE_char]], i64 24, !"_ZTS15NamedBitfields2", [[TYPE_char]], i64 0, i64 1, [[TYPE_char]], i64 1, i64 1, [[TYPE_char]], i64 2, i64 1, [[TYPE_int]], i64 3, i64 4, [[TYPE_int]], i64 3, i64 4, [[TYPE_char]], i64 4, i64 1, [[TYPE_double]], i64 8, i64 8, [[TYPE_int]], i64 16, i64 4}
// CHECK-NEW-DAG: [[TAG_NamedBitfields3]] = !{[[TYPE_NamedBitfields3:!.+]], [[TYPE_NamedBitfields3]], i64 0, i64 16}
// CHECK-NEW-DAG: [[TYPE_NamedBitfields3]] = !{[[TYPE_char]], i64 16, !"_ZTS15NamedBitfields3", [[TYPE_int]], i64 1, i64 4, [[TYPE_int]], i64 2, i64 4, [[TYPE_double]], i64 8, i64 8}
