// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// On a big-endian target a value smaller than its slot is at the high end of
// the slot.

// An int is 4 bytes into its 8-byte general register slot and its 8-byte
// stack slot.
int va_int(__builtin_va_list args) {
  return __builtin_va_arg(args, int);
}

// LLVM-LABEL: define {{.*}} i32 @va_int(
// LLVM:   getelementptr inbounds nuw %struct.__va_list, ptr %{{.*}}, i32 0, i32 3
// LLVM:   %[[GR_TOP:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[SLOT:.*]] = getelementptr i8, ptr %[[GR_TOP]], i64 %{{.*}}
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[SLOT]], i64 4
// OGCG:  %[[SLOT:.*]] = getelementptr inbounds i8, ptr %[[GR_TOP]], i32 %{{.*}}
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[SLOT]], i64 4
// LLVM:   %[[STACK_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[STACK:.*]] = load ptr, ptr %[[STACK_P]], align 8
// LLVM:   %[[NEW_STACK:.*]] = getelementptr {{.*}}i8, ptr %[[STACK]], i64 8
// LLVM:   store ptr %[[NEW_STACK]], ptr %[[STACK_P]], align 8
// LLVMCIR: %[[STACK_ADDR:.*]] = getelementptr i8, ptr %[[STACK]], i64 4
// OGCG:  %[[STACK_ADDR:.*]] = getelementptr inbounds i8, ptr %[[STACK]], i64 4
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK_ADDR]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK_ADDR]], %{{.*}} ]
// LLVM:   %{{.*}} = load i32, ptr %[[ADDR]], align 4

// An 8-byte vector is 8 bytes into its 16-byte vector register slot and fills
// its stack slot.
typedef float V2F __attribute__((vector_size(8)));

V2F va_v2f(__builtin_va_list args) {
  return __builtin_va_arg(args, V2F);
}

// LLVM-LABEL: define {{.*}} <2 x float> @va_v2f(
// LLVM:   getelementptr inbounds nuw %struct.__va_list, ptr %{{.*}}, i32 0, i32 4
// LLVM:   %[[VR_TOP:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[SLOT:.*]] = getelementptr i8, ptr %[[VR_TOP]], i64 %{{.*}}
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[SLOT]], i64 8
// OGCG:  %[[SLOT:.*]] = getelementptr inbounds i8, ptr %[[VR_TOP]], i32 %{{.*}}
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[SLOT]], i64 8
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 8
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   %{{.*}} = load <2 x float>, ptr %[[ADDR]], align 8

// Each member of an HFA is at the high end of its 16-byte vector register
// slot.  On the stack the HFA is laid out as in memory.
struct HFA2 {
  float a, b;
};

struct HFA2 va_hfa2(__builtin_va_list args) {
  return __builtin_va_arg(args, struct HFA2);
}

// LLVM-LABEL: define {{.*}} %struct.HFA2 @va_hfa2(
// LLVM:   %[[VR_TOP:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[SLOT:.*]] = getelementptr i8, ptr %[[VR_TOP]], i64 %{{.*}}
// OGCG:  %[[SLOT:.*]] = getelementptr inbounds i8, ptr %[[VR_TOP]], i32 %{{.*}}
// LLVMCIR: %[[SRC0:.*]] = getelementptr i8, ptr %[[SLOT]], i64 12
// OGCG:  %[[SRC0:.*]] = getelementptr inbounds i8, ptr %[[SLOT]], i64 12
// LLVM:   load float, ptr %[[SRC0]], align 4
// LLVMCIR: %[[SRC1:.*]] = getelementptr i8, ptr %[[SLOT]], i64 28
// OGCG:  %[[SRC1:.*]] = getelementptr inbounds i8, ptr %[[SLOT]], i64 28
// LLVM:   load float, ptr %[[SRC1]], align 4
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 8
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %{{.*}}, %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[ADDR]], i64 8, i1 false)

// A single-member HFA is read in place from the high end of its register slot.
struct HFA1 {
  float a;
};

struct HFA1 va_hfa1(__builtin_va_list args) {
  return __builtin_va_arg(args, struct HFA1);
}

// LLVM-LABEL: define {{.*}} %struct.HFA1 @va_hfa1(
// LLVM:   %[[VR_TOP:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[SLOT:.*]] = getelementptr i8, ptr %[[VR_TOP]], i64 %{{.*}}
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[SLOT]], i64 12
// OGCG:  %[[SLOT:.*]] = getelementptr inbounds i8, ptr %[[VR_TOP]], i32 %{{.*}}
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[SLOT]], i64 12
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[ADDR]], i64 4, i1 false)

// A small record that is not an HFA is read from the start of its slot.
struct Small {
  char c;
};

struct Small va_small(__builtin_va_list args) {
  return __builtin_va_arg(args, struct Small);
}

// LLVM-LABEL: define {{.*}} i64 @va_small(
// LLVM:   %[[GR_TOP:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[GR_TOP]], i64 %{{.*}}
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[GR_TOP]], i32 %{{.*}}
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.*}}, ptr align 8 %[[ADDR]], i64 1, i1 false)
