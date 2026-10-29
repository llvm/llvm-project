// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

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

// CIR-LABEL: cir.func {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// CIR:         %[[NT_THIS:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsNonTrivial>>
// CIR:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CIR-NEXT:    %[[NT_I_DST:.*]] = cir.get_member %[[NT_THIS]][2] {name = "i"}
// CIR-NEXT:    %[[NT_OTHER_I:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsNonTrivial>>
// CIR-NEXT:    %[[NT_I_SRC:.*]] = cir.get_member %[[NT_OTHER_I]][2] {name = "i"}
// CIR-NEXT:    %[[NT_SIZE_IJ:.*]] = cir.const #cir.int<12> : !u64i
// CIR-NEXT:    %[[NT_I_SRC_VOID:.*]] = cir.cast bitcast %[[NT_I_SRC]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    %[[NT_I_DST_VOID:.*]] = cir.cast bitcast %[[NT_I_DST]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[NT_SIZE_IJ]] bytes from %[[NT_I_SRC_VOID]] align(4) to %[[NT_I_DST_VOID]] align(4)
// CIR:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CIR-NEXT:    %[[NT_K_DST:.*]] = cir.get_member %[[NT_THIS]][5] {name = "k"}
// CIR-NEXT:    %[[NT_OTHER_K:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsNonTrivial>>
// CIR-NEXT:    %[[NT_K_SRC:.*]] = cir.get_member %[[NT_OTHER_K]][5] {name = "k"}
// CIR-NEXT:    %[[NT_SIZE_KLM:.*]] = cir.const #cir.int<2> : !u64i
// CIR-NEXT:    %[[NT_K_SRC_VOID:.*]] = cir.cast bitcast %[[NT_K_SRC]] : !cir.ptr<!u16i> -> !cir.ptr<!void>
// CIR-NEXT:    %[[NT_K_DST_VOID:.*]] = cir.cast bitcast %[[NT_K_DST]] : !cir.ptr<!u16i> -> !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[NT_SIZE_KLM]] bytes from %[[NT_K_SRC_VOID]] align(1) to %[[NT_K_DST_VOID]] align(1)
// CIR:         cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// LLVM:         [[NT_THIS:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[NT_OTHER0:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(ptr {{.*}} [[NT_THIS]], ptr {{.*}} [[NT_OTHER0]])
// LLVM-NEXT:    [[NT_I_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_THIS]], i32 0, i32 2
// LLVM-NEXT:    [[NT_OTHER_I:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[NT_I_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_OTHER_I]], i32 0, i32 2
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[NT_I_DST]], ptr align 4 [[NT_I_SRC]], i64 12, i1 false)
// LLVM:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// LLVM-NEXT:    [[NT_K_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_THIS]], i32 0, i32 5
// LLVM-NEXT:    [[NT_OTHER_K:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[NT_K_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_OTHER_K]], i32 0, i32 5
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 1 [[NT_K_DST]], ptr align 1 [[NT_K_SRC]], i64 2, i1 false)
// LLVM:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// LLVM:         ret ptr

// OGCG-LABEL: define {{.*}} @_ZN18ContainsNonTrivialaSERKS_(
// OGCG:         [[NT_THIS:%.*]] = load ptr, ptr %this.addr
// OGCG:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(ptr {{.*}} [[NT_THIS]],
// OGCG-NEXT:    [[NT_I_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_THIS]], i32 0, i32 1
// OGCG-NEXT:    [[NT_OTHER_I:%.*]] = load ptr, ptr %.addr
// OGCG-NEXT:    [[NT_I_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_OTHER_I]], i32 0, i32 1
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[NT_I_DST]], ptr align 4 [[NT_I_SRC]], i64 12, i1 false)
// OGCG:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// OGCG-NEXT:    [[NT_K_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_THIS]], i32 0, i32 4
// OGCG-NEXT:    [[NT_OTHER_K:%.*]] = load ptr, ptr %.addr
// OGCG-NEXT:    [[NT_K_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsNonTrivial, ptr [[NT_OTHER_K]], i32 0, i32 4
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 1 [[NT_K_DST]], ptr align 1 [[NT_K_SRC]], i64 2, i1 false)
// OGCG:         call {{.*}} @_ZN21HasNonTrivialAssignOpaSERKS_(
// OGCG:         ret ptr [[NT_THIS]]
ContainsNonTrivial &
ContainsNonTrivial::operator=(const ContainsNonTrivial &) = default;

struct Trivial {
  int i;
  int *j;
  double k;
  int l[3];
};

// Trivial assignment operators are still called explicitly by CIR. Classic
// codegen never emits the trivial operator and copies the whole object (40
// bytes) inline.

// CIR-LABEL: cir.func {{.*}} @_Z11copyTrivialR7TrivialS0_(
// CIR:         %[[CT_B:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_Trivial>>, !cir.ptr<!rec_Trivial>
// CIR-NEXT:    %[[CT_A:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_Trivial>>, !cir.ptr<!rec_Trivial>
// CIR-NEXT:    cir.call @_ZN7TrivialaSERKS_(%[[CT_A]], %[[CT_B]])
// CIR-NEXT:    cir.return

// LLVM-LABEL: define {{.*}} @_Z11copyTrivialR7TrivialS0_(
// LLVM:         [[CT_B:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[CT_A:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    call {{.*}} @_ZN7TrivialaSERKS_(ptr {{.*}} [[CT_A]], ptr {{.*}} [[CT_B]])
// LLVM-NEXT:    ret void

// OGCG-LABEL: define {{.*}} @_Z11copyTrivialR7TrivialS0_(
// OGCG:         [[CT_B:%.*]] = load ptr, ptr %{{.*}}
// OGCG-NEXT:    [[CT_A:%.*]] = load ptr, ptr %{{.*}}
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[CT_A]], ptr align 8 [[CT_B]], i64 40, i1 false)
// OGCG-NEXT:    ret void
void copyTrivial(Trivial &a, Trivial &b) { a = b; }

// The operator that CIR calls copies all fields in a single memcpy. Note that
// tail padding bytes are not included: sizeof(Trivial) is 40, the memcpy is 36.

// CIR-LABEL: cir.func {{.*}} @_ZN7TrivialaSERKS_(
// CIR:         %[[TR_THIS:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!rec_Trivial>>
// CIR-NEXT:    %[[TR_DST:.*]] = cir.get_member %[[TR_THIS]][0] {name = "i"}
// CIR-NEXT:    %[[TR_OTHER:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_Trivial>>
// CIR-NEXT:    %[[TR_SRC:.*]] = cir.get_member %[[TR_OTHER]][0] {name = "i"}
// CIR-NEXT:    %[[TR_SIZE:.*]] = cir.const #cir.int<36> : !u64i
// CIR-NEXT:    %[[TR_SRC_VOID:.*]] = cir.cast bitcast %[[TR_SRC]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    %[[TR_DST_VOID:.*]] = cir.cast bitcast %[[TR_DST]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[TR_SIZE]] bytes from %[[TR_SRC_VOID]] align(8) to %[[TR_DST_VOID]] align(8)
// CIR-NEXT:    cir.store %[[TR_THIS]], %{{.*}}
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN7TrivialaSERKS_(
// LLVM:         [[TR_THIS:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[TR_DST:%.*]] = getelementptr inbounds nuw %struct.Trivial, ptr [[TR_THIS]], i32 0, i32 0
// LLVM-NEXT:    [[TR_OTHER:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[TR_SRC:%.*]] = getelementptr inbounds nuw %struct.Trivial, ptr [[TR_OTHER]], i32 0, i32 0
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[TR_DST]], ptr align 8 [[TR_SRC]], i64 36, i1 false)
// LLVM:         ret ptr

struct ContainsTrivial {
  Trivial t1;
  Trivial t2;
  ContainsTrivial &operator=(const ContainsTrivial &);
};

// CIR assigns fields of trivial class type by calling their operator=. Classic
// codegen merges both fields into a single memcpy of the whole object.

// CIR-LABEL: cir.func {{.*}} @_ZN15ContainsTrivialaSERKS_(
// CIR:         %[[CTR_THIS:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsTrivial>>
// CIR-NEXT:    %[[CTR_T1_DST:.*]] = cir.get_member %[[CTR_THIS]][0] {name = "t1"}
// CIR-NEXT:    %[[CTR_OTHER1:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsTrivial>>
// CIR-NEXT:    %[[CTR_T1_SRC:.*]] = cir.get_member %[[CTR_OTHER1]][0] {name = "t1"}
// CIR-NEXT:    cir.call @_ZN7TrivialaSERKS_(%[[CTR_T1_DST]], %[[CTR_T1_SRC]])
// CIR-NEXT:    %[[CTR_T2_DST:.*]] = cir.get_member %[[CTR_THIS]][1] {name = "t2"}
// CIR-NEXT:    %[[CTR_OTHER2:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsTrivial>>
// CIR-NEXT:    %[[CTR_T2_SRC:.*]] = cir.get_member %[[CTR_OTHER2]][1] {name = "t2"}
// CIR-NEXT:    cir.call @_ZN7TrivialaSERKS_(%[[CTR_T2_DST]], %[[CTR_T2_SRC]])
// CIR-NOT:     cir.libc.memcpy
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN15ContainsTrivialaSERKS_(
// LLVM:         [[CTR_THIS:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[CTR_T1_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivial, ptr [[CTR_THIS]], i32 0, i32 0
// LLVM-NEXT:    [[CTR_OTHER1:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[CTR_T1_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivial, ptr [[CTR_OTHER1]], i32 0, i32 0
// LLVM-NEXT:    call {{.*}} @_ZN7TrivialaSERKS_(ptr {{.*}} [[CTR_T1_DST]], ptr {{.*}} [[CTR_T1_SRC]])
// LLVM-NEXT:    [[CTR_T2_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivial, ptr [[CTR_THIS]], i32 0, i32 1
// LLVM-NEXT:    [[CTR_OTHER2:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[CTR_T2_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivial, ptr [[CTR_OTHER2]], i32 0, i32 1
// LLVM-NEXT:    call {{.*}} @_ZN7TrivialaSERKS_(ptr {{.*}} [[CTR_T2_DST]], ptr {{.*}} [[CTR_T2_SRC]])
// LLVM-NOT:     call void @llvm.memcpy
// LLVM:         ret ptr

// OGCG-LABEL: define {{.*}} @_ZN15ContainsTrivialaSERKS_(
// OGCG:         [[CTR_THIS:%.*]] = load ptr, ptr %this.addr
// OGCG-NEXT:    [[CTR_T1_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivial, ptr [[CTR_THIS]], i32 0, i32 0
// OGCG-NEXT:    [[CTR_OTHER:%.*]] = load ptr, ptr %.addr
// OGCG-NEXT:    [[CTR_T1_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivial, ptr [[CTR_OTHER]], i32 0, i32 0
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[CTR_T1_DST]], ptr align 8 [[CTR_T1_SRC]], i64 80, i1 false)
// OGCG-NEXT:    ret ptr [[CTR_THIS]]
ContainsTrivial &ContainsTrivial::operator=(const ContainsTrivial &) = default;

struct ContainsTrivialArray {
  Trivial arr[2];
  ContainsTrivialArray &operator=(const ContainsTrivialArray &);
};

// CIR-LABEL: cir.func {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// CIR:         %[[ARR_THIS:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsTrivialArray>>
// CIR-NEXT:    %[[ARR_DST:.*]] = cir.get_member %[[ARR_THIS]][0] {name = "arr"}
// CIR-NEXT:    %[[ARR_DST_VOID:.*]] = cir.cast bitcast %[[ARR_DST]] : {{.*}} -> !cir.ptr<!void>
// CIR-NEXT:    %[[ARR_OTHER:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_ContainsTrivialArray>>
// CIR-NEXT:    %[[ARR_SRC:.*]] = cir.get_member %[[ARR_OTHER]][0] {name = "arr"}
// CIR-NEXT:    %[[ARR_SRC_VOID:.*]] = cir.cast bitcast %[[ARR_SRC]] : {{.*}} -> !cir.ptr<!void>
// CIR-NEXT:    %[[ARR_SIZE:.*]] = cir.const #cir.int<80> : !u64i
// CIR-NEXT:    cir.libc.memcpy %[[ARR_SIZE]] bytes from %[[ARR_SRC_VOID]] align(8) to %[[ARR_DST_VOID]] align(8)
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// LLVM:         [[ARR_THIS:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[ARR_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivialArray, ptr [[ARR_THIS]], i32 0, i32 0
// LLVM-NEXT:    [[ARR_OTHER:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[ARR_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivialArray, ptr [[ARR_OTHER]], i32 0, i32 0
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ARR_DST]], ptr align 8 [[ARR_SRC]], i64 80, i1 false)
// LLVM:         ret ptr

// OGCG-LABEL: define {{.*}} @_ZN20ContainsTrivialArrayaSERKS_(
// OGCG:         [[ARR_THIS:%.*]] = load ptr, ptr %this.addr
// OGCG-NEXT:    [[ARR_DST:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivialArray, ptr [[ARR_THIS]], i32 0, i32 0
// OGCG-NEXT:    [[ARR_OTHER:%.*]] = load ptr, ptr %.addr
// OGCG-NEXT:    [[ARR_SRC:%.*]] = getelementptr inbounds nuw %struct.ContainsTrivialArray, ptr [[ARR_OTHER]], i32 0, i32 0
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[ARR_DST]], ptr align 8 [[ARR_SRC]], i64 80, i1 false)
// OGCG-NEXT:    ret ptr [[ARR_THIS]]
ContainsTrivialArray &
ContainsTrivialArray::operator=(const ContainsTrivialArray &) = default;

// FIXME: CIR zeroes the temporary member by member (leaving its padding
// uninitialized) whereas OGCG memsets
struct HasPadding {
  int a;
  long b[32];
  int c;
};

// CIR-LABEL: cir.func {{.*}} @_Z11resetPaddedP10HasPadding(
// CIR:    cir.call @_ZN10HasPaddingaSEOS_(

// CIR-LABEL: cir.func {{.*}} @_ZN10HasPaddingaSEOS_(
// CIR:         %[[HP_THIS:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!rec_HasPadding>>
// CIR-NEXT:    %[[HP_DST:.*]] = cir.get_member %[[HP_THIS]][0] {name = "a"}
// CIR-NEXT:    %[[HP_OTHER:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_HasPadding>>
// CIR-NEXT:    %[[HP_SRC:.*]] = cir.get_member %[[HP_OTHER]][0] {name = "a"}
// CIR-NEXT:    %[[HP_SIZE:.*]] = cir.const #cir.int<268> : !u64i
// CIR-NEXT:    %[[HP_SRC_VOID:.*]] = cir.cast bitcast %[[HP_SRC]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    %[[HP_DST_VOID:.*]] = cir.cast bitcast %[[HP_DST]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[HP_SIZE]] bytes from %[[HP_SRC_VOID]] align(8) to %[[HP_DST_VOID]] align(8)
// CIR-NEXT:    cir.store %[[HP_THIS]], %{{.*}}
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_Z11resetPaddedP10HasPadding(
// LLVM:         [[RP_TMP:%.*]] = alloca %struct.HasPadding, align 8
// LLVM:         [[RP_P:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    call {{.*}} @_ZN10HasPaddingaSEOS_(ptr {{.*}} [[RP_P]], ptr {{.*}} [[RP_TMP]])

// LLVM-LABEL: define {{.*}} @_ZN10HasPaddingaSEOS_(
// LLVM:         [[HP_THIS:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[HP_DST:%.*]] = getelementptr inbounds nuw %struct.HasPadding, ptr [[HP_THIS]], i32 0, i32 0
// LLVM-NEXT:    [[HP_OTHER:%.*]] = load ptr, ptr {{.*}}
// LLVM-NEXT:    [[HP_SRC:%.*]] = getelementptr inbounds nuw %struct.HasPadding, ptr [[HP_OTHER]], i32 0, i32 0
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[HP_DST]], ptr align 8 [[HP_SRC]], i64 268, i1 false)
// LLVM:         ret ptr

// OGCG-LABEL: define {{.*}} @_Z11resetPaddedP10HasPadding(
// OGCG:         [[RP_TMP:%.*]] = alloca %struct.HasPadding, align 8
// OGCG:         call void @llvm.memset.p0.i64(ptr align 8 [[RP_TMP]], i8 0, i64 272, i1 false)
// OGCG:         [[RP_P:%.*]] = load ptr, ptr %p.addr
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RP_P]], ptr align 8 [[RP_TMP]], i64 272, i1 false)
// OGCG-NEXT:    ret void
// OGCG-NOT:   call {{.*}} @_ZN10HasPaddingaSEOS_(
void resetPadded(HasPadding *p) { *p = {}; }
