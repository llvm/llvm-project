// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Runs of adjacent trivially copyable fields in an implicitly defined copy
// assignment operator are merged into a single memcpy, as classic codegen does.

struct HasNonTrivialAssignOp {
  HasNonTrivialAssignOp &operator=(const HasNonTrivialAssignOp &);
};

struct ContainsNonTrivial {
  HasNonTrivialAssignOp start;
  int i;
  int *j;
  HasNonTrivialAssignOp middle;
  int k : 4;
  int l : 4;
  int m : 4;
  HasNonTrivialAssignOp end;
  ContainsNonTrivial &operator=(const ContainsNonTrivial &);
};

// Fields i and j are merged into one memcpy, k, l and m (bit-fields) into
// another. The non-trivial members are assigned with explicit calls in between.
// CIR-LABEL: cir.func {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// CIR:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CIR:         %[[SIZE_IJ:.*]] = cir.const #cir.int<12> : !u64i
// CIR:         cir.libc.memcpy %[[SIZE_IJ]] bytes from
// CIR:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CIR:         %[[SIZE_KLM:.*]] = cir.const #cir.int<2> : !u64i
// CIR:         cir.libc.memcpy %[[SIZE_KLM]] bytes from
// CIR:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CIR:         cir.return
ContainsNonTrivial &
ContainsNonTrivial::operator=(const ContainsNonTrivial &) = default;

// LLVM-LABEL: define {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// LLVM:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 12, i1 false)
// LLVM:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 2, i1 false)
// LLVM:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(

// OGCG-LABEL: define {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// OGCG:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// OGCG:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 12, i1 false)
// OGCG:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// OGCG:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 2, i1 false)
// OGCG:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(

struct Trivial {
  int i;
  int *j;
  double k;
  int l[3];
};

// Trivial assignment operators are still called explicitly.
// CIR-LABEL: cir.func {{.*}} @_Z11copyTrivialR7TrivialS0_(
// CIR:         cir.call @_ZN7TrivialaSERKS_(
void copyTrivial(Trivial &a, Trivial &b) { a = b; }

// CIR emits the call; the operator body copies only up to the end of the last
// field (36 bytes). Classic codegen never emits the trivial operator and copies
// the whole object (40 bytes) inline.
// LLVM-LABEL: define {{.*}} @_Z11copyTrivialR7TrivialS0_(
// LLVM:         call {{.*}} @_ZN7TrivialaSERKS_(
// LLVM-LABEL: define {{.*}} @_ZN7TrivialaSERKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 36, i1 false)

// OGCG-LABEL: define {{.*}} @_Z11copyTrivialR7TrivialS0_(
// OGCG-NOT:     call {{.*}} @_ZN7TrivialaSERKS_(
// OGCG:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 40, i1 false)

// All fields are merged into one memcpy. Note that tail padding bytes are not
// included: sizeof(Trivial) is 40.
// CIR-LABEL: cir.func {{.*}} @_ZN7TrivialaSERKS_(
// CIR:         %[[SIZE:.*]] = cir.const #cir.int<36> : !u64i
// CIR-NEXT:    cir.cast bitcast
// CIR-NEXT:    cir.cast bitcast
// CIR-NEXT:    cir.libc.memcpy %[[SIZE]] bytes from
// CIR:         cir.return

struct ContainsTrivial {
  Trivial t1;
  Trivial t2;
  ContainsTrivial &operator=(const ContainsTrivial &);
};

// Fields of trivial class type are assigned by calling their operator=.
// CIR-LABEL: cir.func {{.*}} @_ZN15ContainsTrivialaSERKS_(
// CIR:         cir.call @_ZN7TrivialaSERKS_(
// CIR:         cir.call @_ZN7TrivialaSERKS_(
// CIR-NOT:     cir.libc.memcpy
// CIR:         cir.return
ContainsTrivial &ContainsTrivial::operator=(const ContainsTrivial &) = default;

// LLVM-LABEL: define {{.*}} @_ZN15ContainsTrivialaSERKS_(
// LLVM:         call {{.*}} @_ZN7TrivialaSERKS_(
// LLVM:         call {{.*}} @_ZN7TrivialaSERKS_(
// LLVM-NOT:     call void @llvm.memcpy

// Classic codegen merges both fields into a single memcpy of the whole object.
// OGCG-LABEL: define {{.*}} @_ZN15ContainsTrivialaSERKS_(
// OGCG-NOT:     call {{.*}} @_ZN7TrivialaSERKS_(
// OGCG:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 80, i1 false)

struct ContainsTrivialArray {
  Trivial arr[2];
  ContainsTrivialArray &operator=(const ContainsTrivialArray &);
};

// An array of trivial class types is copied with a memcpy.
// CIR-LABEL: cir.func {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// CIR:         %[[SIZE_ARR:.*]] = cir.const #cir.int<80> : !u64i
// CIR:         cir.libc.memcpy %[[SIZE_ARR]] bytes from
// CIR:         cir.return
ContainsTrivialArray &
ContainsTrivialArray::operator=(const ContainsTrivialArray &) = default;

// LLVM-LABEL: define {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 80, i1 false)

// OGCG-LABEL: define {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// OGCG:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 80, i1 false)
