// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | \
// RUN:   FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=LLVM

// Padding after `c` and after `arr`.
struct Padded {
  char c;
  long l;
  int arr[3];
  void *p;
};

void copy_assign(Padded &a, const Padded &b) { a = b; }

// CIR-LABEL: cir.func {{.*}} @_Z11copy_assignR6PaddedRKS_(
// CIR-NOT:     cir.call @_ZN6PaddedaSERKS_
// CIR:         cir.copy {{.*}} : !cir.ptr<!rec_Padded>
// CIR-NOT:     cir.call @_ZN6PaddedaSERKS_

// LLVM-LABEL: define {{.*}} @_Z11copy_assignR6PaddedRKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 40, i1 false)

void move_assign(Padded &a, Padded &&b) { a = static_cast<Padded &&>(b); }

// CIR-LABEL: cir.func {{.*}} @_Z11move_assignR6PaddedOS_(
// CIR-NOT:     cir.call @_ZN6PaddedaSEOS_
// CIR:         cir.copy {{.*}} : !cir.ptr<!rec_Padded>

// LLVM-LABEL: define {{.*}} @_Z11move_assignR6PaddedOS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 40, i1 false)

// The operator's result is its left-hand side.
void chained(Padded &a, Padded &b, const Padded &c) { a = b = c; }

// CIR-LABEL: cir.func {{.*}} @_Z7chainedR6PaddedS0_RKS_(
// CIR:         %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_Padded>>, !cir.ptr<!rec_Padded>
// CIR:         %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_Padded>>, !cir.ptr<!rec_Padded>
// CIR:         cir.copy %[[C]] to %[[B]] :
// CIR:         %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_Padded>>, !cir.ptr<!rec_Padded>
// CIR:         cir.copy %[[B]] to %[[A]] :

// LLVM-LABEL: define {{.*}} @_Z7chainedR6PaddedS0_RKS_(
// LLVM-COUNT-2: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 40, i1 false)

// Not a POD for layout purposes, so a derived class may place members in its
// tail padding, so only its 5 data bytes are copied.
struct TailPadded {
  TailPadded() {}
  int x;
  char c;
};

void tail_padded(TailPadded &a, const TailPadded &b) { a = b; }

// CIR-LABEL: cir.func {{.*}} @_Z11tail_paddedR10TailPaddedRKS_(
// CIR:         cir.copy {{.*}} skip_tail_padding : !cir.ptr<!rec_TailPadded>

// LLVM-LABEL: define {{.*}} @_Z11tail_paddedR10TailPaddedRKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 5, i1 false)

// A union is copied whole.
union Union {
  int i;
  char c[7];
};

void union_assign(Union &a, const Union &b) { a = b; }

// CIR-LABEL: cir.func {{.*}} @_Z12union_assignR5UnionRKS_(
// CIR-NOT:     skip_tail_padding
// CIR:         cir.copy {{.*}} : !cir.ptr<!rec_Union>

// LLVM-LABEL: define {{.*}} @_Z12union_assignR5UnionRKS_(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)

// A user-provided assignment operator is still called.
struct UserAssign {
  int x;
  UserAssign &operator=(const UserAssign &);
};

void user_assign(UserAssign &a, const UserAssign &b) { a = b; }

// CIR-LABEL: cir.func {{.*}} @_Z11user_assignR10UserAssignRKS_(
// CIR:         cir.call @_ZN10UserAssignaSERKS_(

// LLVM-LABEL: define {{.*}} @_Z11user_assignR10UserAssignRKS_(
// LLVM:         call {{.*}} @_ZN10UserAssignaSERKS_(
