// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

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
// CHECK-LABEL: cir.func {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// CHECK:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CHECK:         %[[SIZE_IJ:.*]] = cir.const #cir.int<12> : !u64i
// CHECK:         cir.libc.memcpy %[[SIZE_IJ]] bytes from
// CHECK:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CHECK:         %[[SIZE_KLM:.*]] = cir.const #cir.int<2> : !u64i
// CHECK:         cir.libc.memcpy %[[SIZE_KLM]] bytes from
// CHECK:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CHECK:         cir.return
ContainsNonTrivial &
ContainsNonTrivial::operator=(const ContainsNonTrivial &) = default;

struct Trivial {
  int i;
  int *j;
  double k;
  int l[3];
};

// Trivial assignment operators are still called explicitly.
// CHECK-LABEL: cir.func {{.*}} @_Z11copyTrivialR7TrivialS0_(
// CHECK:         cir.call @_ZN7TrivialaSERKS_(
void copyTrivial(Trivial &a, Trivial &b) { a = b; }

// All fields are merged into one memcpy. Note that tail padding bytes are not
// included: sizeof(Trivial) is 40.
// CHECK-LABEL: cir.func {{.*}} @_ZN7TrivialaSERKS_(
// CHECK:         %[[SIZE:.*]] = cir.const #cir.int<36> : !u64i
// CHECK-NEXT:    cir.cast bitcast
// CHECK-NEXT:    cir.cast bitcast
// CHECK-NEXT:    cir.libc.memcpy %[[SIZE]] bytes from
// CHECK:         cir.return

struct ContainsTrivial {
  Trivial t1;
  Trivial t2;
  ContainsTrivial &operator=(const ContainsTrivial &);
};

// Fields of trivial class type are assigned by calling their operator=.
// CHECK-LABEL: cir.func {{.*}} @_ZN15ContainsTrivialaSERKS_(
// CHECK:         cir.call @_ZN7TrivialaSERKS_(
// CHECK:         cir.call @_ZN7TrivialaSERKS_(
// CHECK-NOT:     cir.libc.memcpy
// CHECK:         cir.return
ContainsTrivial &ContainsTrivial::operator=(const ContainsTrivial &) = default;

struct ContainsTrivialArray {
  Trivial arr[2];
  ContainsTrivialArray &operator=(const ContainsTrivialArray &);
};

// An array of trivial class types is copied with a memcpy.
// CHECK-LABEL: cir.func {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// CHECK:         %[[SIZE_ARR:.*]] = cir.const #cir.int<80> : !u64i
// CHECK:         cir.libc.memcpy %[[SIZE_ARR]] bytes from
// CHECK:         cir.return
ContainsTrivialArray &
ContainsTrivialArray::operator=(const ContainsTrivialArray &) = default;
