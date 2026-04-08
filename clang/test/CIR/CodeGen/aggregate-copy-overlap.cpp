// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Test that emitAggregateCopy uses the data size (excluding tail padding)
// when copying potentially-overlapping subobjects, and uses full type size
// otherwise.

struct Base { int x; };

struct HasPadding : Base {
  char c;
  // sizeof(HasPadding) = 8 (4 for x, 1 for c, 3 tail padding)
  // data size = 5
};

struct VBase { int v; };

// Outer has a virtual base, so its nvsize (14) is smaller than its full
// sizeof (24). Because [[no_unique_address]] HasPadding extends beyond
// nvsize (offset 8 + sizeof 8 = 16 > 14), getOverlapForFieldInit returns
// MayOverlap, and emitAggregateCopy must use the data size (5) instead of
// the full sizeof (8).
struct Outer : virtual VBase {
  [[no_unique_address]] HasPadding hp;
  char extra;
  Outer(const HasPadding &hp, char e) : hp(hp), extra(e) {}
};

// With virtual bases, only the C1 (complete) constructor is emitted.
// CIR-LABEL: cir.func {{.*}} @_ZN5OuterC1ERK10HasPaddingc(
// CIR:         cir.copy %{{.+}} to %{{.+}} skip_tail_padding : !cir.ptr<!rec_HasPadding>

// LLVM-LABEL: define {{.*}} void @_ZN5OuterC1ERK10HasPaddingc(
// LLVM:         %[[GEP:.*]] = getelementptr %struct.Outer, ptr %{{.+}}, i32 0, i32 1
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr %[[GEP]], ptr %{{.+}}, i64 5, i1 false)

// OGCG-LABEL: define {{.*}} void @_ZN5OuterC1ERK10HasPaddingc(
// OGCG:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.+}}, i32 0, i32 1
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[GEP]], ptr {{.*}} %{{.+}}, i64 5, i1 false)

void test_overlap(const HasPadding &hp) {
  Outer o(hp, 'x');
}

// NonOverlapping does NOT have [[no_unique_address]], so the copy uses
// cir.copy (full type size) rather than cir.libc.memcpy.
struct NonOverlapping {
  HasPadding hp;
  char extra;
  NonOverlapping(const HasPadding &hp, char e) : hp(hp), extra(e) {}
};

// CIR-LABEL: cir.func {{.*}} @_ZN14NonOverlappingC2ERK10HasPaddingc(
// CIR:         cir.copy %{{.+}} to %{{.+}} : !cir.ptr<!rec_HasPadding>

// LLVM-LABEL: define {{.*}} void @_ZN14NonOverlappingC2ERK10HasPaddingc(
// LLVM:         %[[GEP:.*]] = getelementptr %struct.NonOverlapping, ptr %{{.+}}, i32 0, i32 0
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr %[[GEP]], ptr %{{.+}}, i64 8, i1 false)

// OGCG-LABEL: define {{.*}} void @_ZN14NonOverlappingC2ERK10HasPaddingc(
// OGCG:         %[[GEP:.*]] = getelementptr inbounds nuw %struct.NonOverlapping, ptr %{{.+}}, i32 0, i32 0
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[GEP]], ptr {{.*}} %{{.+}}, i64 8, i1 false)

void test_no_overlap(const HasPadding &hp) {
  NonOverlapping o(hp, 'x');
}
