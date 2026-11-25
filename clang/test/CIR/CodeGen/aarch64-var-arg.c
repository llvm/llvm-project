// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// An int is read from the general register save area through __gr_top and
// __gr_offs, or from the stack through __stack.
int va_int(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  int res = __builtin_va_arg(args, int);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @va_int(
// CIR:   %[[ARGS:.*]] = cir.alloca "args" align(8) : !cir.ptr<!rec___va_list>
// CIR:   cir.va_start %[[ARGS]]
// CIR:   %[[GR_OFFS_P:.*]] = cir.get_member %[[ARGS]][3] {name = "__gr_offs"} : !cir.ptr<!rec___va_list> -> !cir.ptr<!s32i>
// CIR:   %[[GR_OFFS:.*]] = cir.load align(8) %[[GR_OFFS_P]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[ON_STACK:.*]] = cir.cmp ge %[[GR_OFFS]], %[[ZERO]] : !s32i
// CIR:   %[[REG_SIZE:.*]] = cir.const #cir.int<8> : !s32i
// CIR:   %[[NEW_OFFS:.*]] = cir.add %[[GR_OFFS]], %[[REG_SIZE]] : !s32i
// CIR:   %[[IN_REGS:.*]] = cir.ternary(%[[ON_STACK]], true {
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.yield %[[FALSE]] : !cir.bool
// CIR:   }, false {
// CIR:     cir.store align(8) %[[NEW_OFFS]], %[[GR_OFFS_P]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[FITS:.*]] = cir.cmp le %[[NEW_OFFS]], %[[ZERO]] : !s32i
// CIR:     cir.yield %[[FITS]] : !cir.bool
// CIR:   }) : (!cir.bool) -> !cir.bool
// CIR:   %[[ADDR:.*]] = cir.ternary(%[[IN_REGS]], true {
// CIR:     %[[GR_TOP_P:.*]] = cir.get_member %[[ARGS]][1] {name = "__gr_top"} : !cir.ptr<!rec___va_list> -> !cir.ptr<!cir.ptr<!void>>
// CIR:     %[[GR_TOP:.*]] = cir.load %[[GR_TOP_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:     %[[GR_TOP_B:.*]] = cir.cast bitcast %[[GR_TOP]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.*]] = cir.ptr_stride %[[GR_TOP_B]], %[[GR_OFFS]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     %[[STACK_P:.*]] = cir.get_member %[[ARGS]][0] {name = "__stack"} : !cir.ptr<!rec___va_list> -> !cir.ptr<!cir.ptr<!void>>
// CIR:     %[[STACK:.*]] = cir.load %[[STACK_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:     %[[STACK_B:.*]] = cir.cast bitcast %[[STACK]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[STACK_SIZE:.*]] = cir.const #cir.int<8> : !s64i
// CIR:     %[[NEW_STACK:.*]] = cir.ptr_stride %[[STACK_B]], %[[STACK_SIZE]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:     %[[STACK_P_B:.*]] = cir.cast bitcast %[[STACK_P]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.ptr<!u8i>>
// CIR:     cir.store %[[NEW_STACK]], %[[STACK_P_B]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:     cir.yield %[[STACK_B]] : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[ADDR_I:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s32i>
// CIR:   %{{.*}} = cir.load align(8) %[[ADDR_I]] : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: define {{.*}} i32 @va_int(
// LLVM:   %[[GR_OFFS_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS:.*]], i32 0, i32 3
// LLVM:   %[[GR_OFFS:.*]] = load i32, ptr %[[GR_OFFS_P]], align 8
// LLVM:   %{{.*}} = icmp sge i32 %[[GR_OFFS]], 0
// LLVM:   %[[NEW_OFFS:.*]] = add i32 %[[GR_OFFS]], 8
// LLVM:   store i32 %[[NEW_OFFS]], ptr %[[GR_OFFS_P]], align 8
// LLVM:   %{{.*}} = icmp sle i32 %[[NEW_OFFS]], 0
// LLVM:   %[[GR_TOP_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS]], i32 0, i32 1
// LLVM:   %[[GR_TOP:.*]] = load ptr, ptr %[[GR_TOP_P]], align 8
// LLVMCIR: %[[GR_OFFS_EXT:.*]] = sext i32 %[[GR_OFFS]] to i64
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[GR_TOP]], i64 %[[GR_OFFS_EXT]]
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[GR_TOP]], i32 %[[GR_OFFS]]
// LLVM:   %[[STACK_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS]], i32 0, i32 0
// LLVM:   %[[STACK:.*]] = load ptr, ptr %[[STACK_P]], align 8
// LLVMCIR: %[[NEW_STACK:.*]] = getelementptr i8, ptr %[[STACK]], i64 8
// OGCG:  %[[NEW_STACK:.*]] = getelementptr inbounds i8, ptr %[[STACK]], i64 8
// LLVM:   store ptr %[[NEW_STACK]], ptr %[[STACK_P]], align 8
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   %{{.*}} = load i32, ptr %[[ADDR]], align 8

// A double is read from the vector register save area through __vr_top and
// __vr_offs, which advances by a 16-byte register.  A stack slot is 8 bytes.
double va_double(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  double res = __builtin_va_arg(args, double);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @va_double(
// CIR:   %[[VR_OFFS_P:.*]] = cir.get_member %{{.*}}[4] {name = "__vr_offs"} : !cir.ptr<!rec___va_list> -> !cir.ptr<!s32i>
// CIR:   %[[VR_OFFS:.*]] = cir.load align(4) %[[VR_OFFS_P]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[REG_SIZE:.*]] = cir.const #cir.int<16> : !s32i
// CIR:   %{{.*}} = cir.add %[[VR_OFFS]], %[[REG_SIZE]] : !s32i
// CIR:   cir.get_member %{{.*}}[2] {name = "__vr_top"}
// CIR:   cir.get_member %{{.*}}[0] {name = "__stack"}
// CIR:   %{{.*}} = cir.const #cir.int<8> : !s64i
// CIR:   cir.load align(8) %{{.*}} : !cir.ptr<!cir.double>, !cir.double

// LLVM-LABEL: define {{.*}} double @va_double(
// LLVM:   %[[VR_OFFS_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS:.*]], i32 0, i32 4
// LLVM:   %[[VR_OFFS:.*]] = load i32, ptr %[[VR_OFFS_P]], align 4
// LLVM:   %[[NEW_OFFS:.*]] = add i32 %[[VR_OFFS]], 16
// LLVM:   store i32 %[[NEW_OFFS]], ptr %[[VR_OFFS_P]], align 4
// LLVM:   getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS]], i32 0, i32 2
// LLVM:   %[[STACK_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS]], i32 0, i32 0
// LLVM:   %[[STACK:.*]] = load ptr, ptr %[[STACK_P]], align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 8
// LLVM:   %[[ADDR:.*]] = phi ptr
// LLVM:   %{{.*}} = load double, ptr %[[ADDR]], align 8

// The members of an HFA are 16 bytes apart in the vector register save area,
// so they are copied into a temporary where they are contiguous.
struct HFA {
  float a, b, c;
};

struct HFA va_hfa(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct HFA res = __builtin_va_arg(args, struct HFA);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @va_hfa(
// CIR:   %[[VR_OFFS_P:.*]] = cir.get_member %{{.*}}[4] {name = "__vr_offs"}
// CIR:   %[[VR_OFFS:.*]] = cir.load align(4) %[[VR_OFFS_P]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[REG_SIZE:.*]] = cir.const #cir.int<48> : !s32i
// CIR:   %{{.*}} = cir.add %[[VR_OFFS]], %[[REG_SIZE]] : !s32i
// CIR:   %[[HFA_TMP:.*]] = cir.alloca "vaarg.hfa" align(4) : !cir.ptr<!cir.array<!cir.float x 3>>
// CIR:   %[[ADDR:.*]] = cir.ternary(%{{.*}}, true {
// CIR:     %[[VR_TOP_P:.*]] = cir.get_member %{{.*}}[2] {name = "__vr_top"}
// CIR:     %[[VR_TOP:.*]] = cir.load %[[VR_TOP_P]]
// CIR:     %[[VR_TOP_B:.*]] = cir.cast bitcast %[[VR_TOP]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.*]] = cir.ptr_stride %[[VR_TOP_B]], %[[VR_OFFS]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     %[[ELEMS:.*]] = cir.cast array_to_ptrdecay %[[HFA_TMP]] : !cir.ptr<!cir.array<!cir.float x 3>> -> !cir.ptr<!cir.float>
// CIR:     %[[OFF0:.*]] = cir.const #cir.int<0> : !s64i
// CIR:     %[[SRC0:.*]] = cir.ptr_stride %[[REG_ADDR]], %[[OFF0]]
// CIR:     %[[SRC0_F:.*]] = cir.cast bitcast %[[SRC0]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.float>
// CIR:     %[[ELEM0:.*]] = cir.load align(16) %[[SRC0_F]] : !cir.ptr<!cir.float>, !cir.float
// CIR:     %[[IDX0:.*]] = cir.const #cir.int<0> : !s64i
// CIR:     %[[DST0:.*]] = cir.ptr_stride %[[ELEMS]], %[[IDX0]]
// CIR:     cir.store align(4) %[[ELEM0]], %[[DST0]] : !cir.float, !cir.ptr<!cir.float>
// CIR:     %[[OFF1:.*]] = cir.const #cir.int<16> : !s64i
// CIR:     %[[SRC1:.*]] = cir.ptr_stride %[[REG_ADDR]], %[[OFF1]]
// CIR:     %[[SRC1_F:.*]] = cir.cast bitcast %[[SRC1]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.float>
// CIR:     %[[ELEM1:.*]] = cir.load align(16) %[[SRC1_F]] : !cir.ptr<!cir.float>, !cir.float
// CIR:     %[[IDX1:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     %[[DST1:.*]] = cir.ptr_stride %[[ELEMS]], %[[IDX1]]
// CIR:     cir.store align(4) %[[ELEM1]], %[[DST1]] : !cir.float, !cir.ptr<!cir.float>
// CIR:     %[[OFF2:.*]] = cir.const #cir.int<32> : !s64i
// CIR:     %[[SRC2:.*]] = cir.ptr_stride %[[REG_ADDR]], %[[OFF2]]
// CIR:     %[[SRC2_F:.*]] = cir.cast bitcast %[[SRC2]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.float>
// CIR:     %[[ELEM2:.*]] = cir.load align(16) %[[SRC2_F]] : !cir.ptr<!cir.float>, !cir.float
// CIR:     %[[IDX2:.*]] = cir.const #cir.int<2> : !s64i
// CIR:     %[[DST2:.*]] = cir.ptr_stride %[[ELEMS]], %[[IDX2]]
// CIR:     cir.store align(4) %[[ELEM2]], %[[DST2]] : !cir.float, !cir.ptr<!cir.float>
// CIR:     %[[TMP_B:.*]] = cir.cast bitcast %[[HFA_TMP]] : !cir.ptr<!cir.array<!cir.float x 3>> -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[TMP_B]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     cir.get_member %{{.*}}[0] {name = "__stack"}
// CIR:     %{{.*}} = cir.const #cir.int<16> : !s64i
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[ADDR_R:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_HFA>
// CIR:   cir.copy %[[ADDR_R]] align(4) to %{{.*}} align(4) : !cir.ptr<!rec_HFA>

// LLVM-LABEL: define {{.*}} %struct.HFA @va_hfa(
// LLVMCIR: %[[HFA_TMP:.*]] = alloca [3 x float], align 4
// OGCG:  %[[HFA_TMP:.*]] = alloca [3 x float], align 4
// LLVM:   %[[VR_OFFS_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS:.*]], i32 0, i32 4
// LLVM:   %[[VR_OFFS:.*]] = load i32, ptr %[[VR_OFFS_P]], align 4
// LLVM:   %[[NEW_OFFS:.*]] = add i32 %[[VR_OFFS]], 48
// LLVM:   store i32 %[[NEW_OFFS]], ptr %[[VR_OFFS_P]], align 4
// LLVM:   getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS]], i32 0, i32 2
// LLVMCIR: getelementptr i8, ptr %[[REG_ADDR:[0-9]+]], i64 0
// OGCG:  getelementptr inbounds i8, ptr %[[REG_ADDR:[0-9]+]], i64 0
// LLVM:   %[[ELEM0:.*]] = load float, ptr %{{.*}}, align 16
// LLVM:   store float %[[ELEM0]], ptr %{{.*}}, align 4
// LLVMCIR: getelementptr i8, ptr %[[REG_ADDR]], i64 16
// OGCG:  getelementptr inbounds i8, ptr %[[REG_ADDR]], i64 16
// LLVM:   %[[ELEM1:.*]] = load float, ptr %{{.*}}, align 16
// LLVM:   store float %[[ELEM1]], ptr %{{.*}}, align 4
// LLVMCIR: getelementptr i8, ptr %[[REG_ADDR]], i64 32
// OGCG:  getelementptr inbounds i8, ptr %[[REG_ADDR]], i64 32
// LLVM:   %[[ELEM2:.*]] = load float, ptr %{{.*}}, align 16
// LLVM:   store float %[[ELEM2]], ptr %{{.*}}, align 4
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 16
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[HFA_TMP]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[HFA_TMP]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[ADDR]], i64 12, i1 false)

// A 16-byte-aligned value starts at an even general register, so __gr_offs is
// rounded up before it is used.  On the stack, __stack is rounded up.
struct Align16 {
  __int128 a;
};

struct Align16 va_align16(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Align16 res = __builtin_va_arg(args, struct Align16);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @va_align16(
// CIR:   %[[GR_OFFS:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   %[[BUMP:.*]] = cir.const #cir.int<15> : !s32i
// CIR:   %[[BUMPED:.*]] = cir.add %[[GR_OFFS]], %[[BUMP]] : !s32i
// CIR:   %[[MASK:.*]] = cir.const #cir.int<-16> : !s32i
// CIR:   %[[ALIGNED:.*]] = cir.and %[[BUMPED]], %[[MASK]] : !s32i
// CIR:   %[[REG_SIZE:.*]] = cir.const #cir.int<16> : !s32i
// CIR:   %{{.*}} = cir.add %[[ALIGNED]], %[[REG_SIZE]] : !s32i
// CIR:   cir.ternary
// CIR:     cir.get_member %{{.*}}[1] {name = "__gr_top"}
// CIR:     cir.ptr_stride %{{.*}}, %[[ALIGNED]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     cir.get_member %{{.*}}[0] {name = "__stack"}
// CIR:     %[[STACK_MASK:.*]] = cir.const #cir.int<-16> : !s64i
// CIR:     %[[STACK_ALIGNED:.*]] = cir.ptr_mask %{{.*}}, %[[STACK_MASK]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:     %[[STACK_SIZE:.*]] = cir.const #cir.int<16> : !s64i
// CIR:     cir.ptr_stride %[[STACK_ALIGNED]], %[[STACK_SIZE]]
// CIR:     cir.yield %[[STACK_ALIGNED]] : !cir.ptr<!u8i>
// CIR:   cir.copy %{{.*}} align(8) to %{{.*}} align(16) : !cir.ptr<!rec_Align16>

// LLVM-LABEL: define {{.*}} i128 @va_align16(
// LLVM:   %[[GR_OFFS:.*]] = load i32, ptr %{{.*}}, align 8
// LLVM:   %[[BUMPED:.*]] = add i32 %[[GR_OFFS]], 15
// LLVM:   %[[ALIGNED:.*]] = and i32 %[[BUMPED]], -16
// LLVM:   %[[NEW_OFFS:.*]] = add i32 %[[ALIGNED]], 16
// LLVM:   store i32 %[[NEW_OFFS]], ptr %{{.*}}, align 8
// LLVMCIR: %[[ALIGNED_EXT:.*]] = sext i32 %[[ALIGNED]] to i64
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %{{.*}}, i64 %[[ALIGNED_EXT]]
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[ALIGNED]]
// LLVM:   %[[STACK_ALIGNED:.*]] = call ptr @llvm.ptrmask.p0.i64(ptr %{{.*}}, i64 -16)
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK_ALIGNED]], i64 16
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK_ALIGNED]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK_ALIGNED]], %{{.*}} ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.*}}, ptr align 8 %[[ADDR]], i64 16, i1 false)

// A record larger than 16 bytes is passed as a pointer to a copy, so the slot
// holds a pointer to the value.
struct Big {
  long a[5];
};

struct Big va_big(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Big res = __builtin_va_arg(args, struct Big);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @va_big(
// CIR:   cir.get_member %{{.*}}[3] {name = "__gr_offs"}
// CIR:   %{{.*}} = cir.const #cir.int<8> : !s32i
// CIR:   cir.get_member %{{.*}}[0] {name = "__stack"}
// CIR:   %{{.*}} = cir.const #cir.int<8> : !s64i
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[ADDR_P:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!u8i> -> !cir.ptr<!cir.ptr<!rec_Big>>
// CIR:   %[[VAL_P:.*]] = cir.load align(8) %[[ADDR_P]] : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
// CIR:   cir.copy %[[VAL_P]] align(8) to %{{.*}} align(8) : !cir.ptr<!rec_Big>

// LLVM-LABEL: define {{.*}} void @va_big(
// LLVM:   %[[GR_OFFS:.*]] = load i32, ptr %{{.*}}, align 8
// LLVM:   add i32 %[[GR_OFFS]], 8
// LLVM:   %[[STACK_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[STACK:.*]] = load ptr, ptr %[[STACK_P]], align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 8
// LLVM:   %[[ADDR:.*]] = phi ptr
// LLVM:   %[[VAL_P:.*]] = load ptr, ptr %[[ADDR]], align 8
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %[[VAL_P]], i64 40, i1 false)
