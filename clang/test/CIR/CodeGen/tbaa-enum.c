// This is inspired from clang/test/CodeGen/tbaa.c, with both CIR and LLVM checks.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes -relaxed-aliasing
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O0 -disable-llvm-passes
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s

// NO-TBAA-NOT: !tbaa

// CIR: #tbaa[[NYI:.*]] = #cir.tbaa
// CIR: #tbaa[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR: #tbaa[[INT:.*]] = #cir.tbaa_scalar<id = "int", type = !s32i>
// CIR: #tbaa[[LONG_LONG:.*]] = #cir.tbaa_scalar<id = "long long", type = !s64i>
// CIR: #tbaa[[SHORT:.*]] = #cir.tbaa_scalar<id = "short", type = !s16i>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

typedef enum {
  RED_AUTO_32,
  GREEN_AUTO_32,
  BLUE_AUTO_32
} EnumAuto32;

typedef enum {
  RED_AUTO_64,
  GREEN_AUTO_64,
  BLUE_AUTO_64 = 0x100000000ull
} EnumAuto64;

typedef enum : uint16_t {
  RED_16,
  GREEN_16,
  BLUE_16
} Enum16;

typedef enum : uint8_t {
  RED_8,
  GREEN_8,
  BLUE_8
} Enum8;

uint32_t g0(EnumAuto32 *E, uint32_t *val) {
  // CIR-LABEL: cir.func @g0
  // CIR: %[[C5:.*]] = cir.const #cir.int<5> : !s32i
  // CIR: %[[U_C5:.*]] = cir.cast(integral, %[[C5]] : !s32i), !u32i
  // CIR: %[[VAL_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
  // CIR: cir.store %[[U_C5]], %[[VAL_PTR]] : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[INT]])
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[U_C0:.*]] = cir.cast(integral, %[[C0]] : !s32i), !u32i
  // CIR: %[[E_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
  // CIR: cir.store %[[U_C0]], %[[E_PTR]] : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[NYI]])
  // CIR: %[[RET_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u32i>, !u32i tbaa(#tbaa[[INT]])
  // CIR: cir.store %[[RET]], %{{.*}} : !u32i, !cir.ptr<!u32i>

  // LLVM-LABEL: define{{.*}} i32 @g0(
  // LLVM: store i32 5, ptr %{{.*}}, align 4, !tbaa [[TAG_i32:!.*]]
  // LLVM: store i32 0, ptr %{{.*}}, align 4
  // LLVM: load i32, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  *val = 5;
  *E = RED_AUTO_32;
  return *val;
}

uint64_t g1(EnumAuto64 *E, uint64_t *val) {
  // CIR-LABEL: cir.func @g1
  // CIR: %[[C5:.*]] = cir.const #cir.int<5> : !s32i
  // CIR: %[[U_C5:.*]] = cir.cast(integral, %[[C5]] : !s32i), !u64i
  // CIR: %[[VAL_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
  // CIR: cir.store %[[U_C5]], %[[VAL_PTR]] : !u64i, !cir.ptr<!u64i> tbaa(#tbaa[[LONG_LONG]])
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[E_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
  // CIR: cir.store %[[C0]], %[[E_PTR]] : !u64i, !cir.ptr<!u64i> tbaa(#tbaa[[NYI]])
  // CIR: %[[RET_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u64i>, !u64i tbaa(#tbaa[[LONG_LONG]])
  // CIR: cir.store %[[RET]], %{{.*}} : !u64i, !cir.ptr<!u64i>

  // LLVM-LABEL: define{{.*}} i64 @g1(
  // LLVM: store i64 5, ptr %{{.*}}, align 8, !tbaa [[TAG_i64:!.*]]
  // LLVM: store i64 0, ptr %{{.*}}, align 8
  // LLVM: load i64, ptr %{{.*}}, align 8, !tbaa [[TAG_i64]]
  *val = 5;
  *E = RED_AUTO_64;
  return *val;
}

uint16_t g2(Enum16 *E, uint16_t *val) {
  // CIR-LABEL: cir.func @g2
  // CIR: %[[C5:.*]] = cir.const #cir.int<5> : !s32i
  // CIR: %[[U_C5:.*]] = cir.cast(integral, %[[C5]] : !s32i), !u16i
  // CIR: %[[VAL_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u16i>>, !cir.ptr<!u16i>
  // CIR: cir.store %[[U_C5]], %[[VAL_PTR]] : !u16i, !cir.ptr<!u16i> tbaa(#tbaa[[SHORT]])
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !u16i
  // CIR: %[[E_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u16i>>, !cir.ptr<!u16i>
  // CIR: cir.store %[[C0]], %[[E_PTR]] : !u16i, !cir.ptr<!u16i> tbaa(#tbaa[[NYI]])
  // CIR: %[[RET_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u16i>>, !cir.ptr<!u16i>
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u16i>, !u16i tbaa(#tbaa[[SHORT]])
  // CIR: cir.store %[[RET]], %{{.*}} : !u16i, !cir.ptr<!u16i>

  // LLVM-LABEL: define{{.*}} i16 @g2(
  // LLVM: store i16 5, ptr %{{.*}}, align 2, !tbaa [[TAG_i16:!.*]]
  // LLVM: store i16 0, ptr %{{.*}}, align 2
  // LLVM: load i16, ptr %{{.*}}, align 2, !tbaa [[TAG_i16]]
  *val = 5;
  *E = RED_16;
  return *val;
}

uint8_t g3(Enum8 *E, uint8_t *val) {
  // CIR-LABEL: cir.func @g3
  // CIR: %[[C5:.*]] = cir.const #cir.int<5> : !s32i
  // CIR: %[[U_C5:.*]] = cir.cast(integral, %[[C5]] : !s32i), !u8i
  // CIR: %[[VAL_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
  // CIR: cir.store %[[U_C5]], %[[VAL_PTR]] : !u8i, !cir.ptr<!u8i> tbaa(#tbaa[[CHAR]])
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: %[[E_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
  // CIR: cir.store %[[C0]], %[[E_PTR]] : !u8i, !cir.ptr<!u8i> tbaa(#tbaa[[NYI]])
  // CIR: %[[RET_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u8i>, !u8i tbaa(#tbaa[[CHAR]])
  // CIR: cir.store %[[RET]], %{{.*}} : !u8i, !cir.ptr<!u8i>
  

  // LLVM-LABEL: define{{.*}} i8 @g3(
  // LLVM: store i8 5, ptr %{{.*}}, align 1, !tbaa [[TAG_i8:!.*]]
  // LLVM: store i8 0, ptr %{{.*}}, align 1
  // LLVM: load i8, ptr %{{.*}}, align 1, !tbaa [[TAG_i8]]
  *val = 5;
  *E = RED_8;
  return *val;
}

// LLVM: [[TAG_i32]] = !{[[TYPE_i32:!.*]], [[TYPE_i32]], i64 0}
// LLVM: [[TYPE_i32]] = !{!"int", [[TYPE_char:!.*]],
// LLVM: [[TYPE_char]] = !{!"omnipotent char", [[TAG_c_tbaa:!.*]],
// LLVM: [[TAG_c_tbaa]] = !{!"Simple C/C++ TBAA"}
// LLVM: [[TAG_i64]] = !{[[TYPE_i64:!.*]], [[TYPE_i64]], i64 0}
// LLVM: [[TYPE_i64]] = !{!"long long", [[TYPE_char]],
// LLVM: [[TAG_i16]] = !{[[TYPE_i16:!.*]], [[TYPE_i16]], i64 0}
// LLVM: [[TYPE_i16]] = !{!"short", [[TYPE_char]],
// LLVM: [[TAG_i8]] = !{[[TYPE_i8:!.*]], [[TYPE_char]], i64 0}