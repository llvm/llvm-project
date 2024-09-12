// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-int-conversions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-int-conversions -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// GNU extensions
typedef void (*FP)(void);
void *f2(void *a, int b) { return a + b; }
// CIR-LABEL: f2
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!void>, %[[STRIDE]] : !s32i)

// LLVM-LABEL: f2
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[STRIDE]]

// These test the same paths above, just make sure it does not crash.
void *f2_0(void *a, int b) { return &a[b]; }
void *f2_1(void *a, int b) { return (a += b); }
void *f3(int a, void *b) { return a + b; }

void *f3_1(int a, void *b) { return (a += b); }
// CIR-LABEL: @f3_1
// CIR: %[[NEW_PTR:.*]] = cir.ptr_stride
// CIR: cir.cast(ptr_to_int, %[[NEW_PTR]] : !cir.ptr<!void>), !s32i

// LLVM-LABEL: @f3_1
// LLVM: %[[NEW_PTR:.*]] = getelementptr
// LLVM: ptrtoint ptr %[[NEW_PTR]] to i32

void *f4(void *a, int b) { return a - b; }
// CIR-LABEL: f4
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[SUB:.*]] = cir.unary(minus, %[[STRIDE]]) : !s32i, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!void>, %[[SUB]] : !s32i)

// LLVM-LABEL: f4
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: %[[SUB:.*]] = sub i64 0, %[[STRIDE]]
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[SUB]]

// Similar to f4, just make sure it does not crash.
void *f4_1(void *a, int b) { return (a -= b); }

FP f5(FP a, int b) { return a + b; }
// CIR-LABEL: f5
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!cir.func<!void ()>>>, !cir.ptr<!cir.func<!void ()>>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!cir.func<!void ()>>, %[[STRIDE]] : !s32i)

// LLVM-LABEL: f5
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[STRIDE]]

// These test the same paths above, just make sure it does not crash.
FP f5_1(FP a, int b) { return (a += b); }
FP f6(int a, FP b) { return a + b; }
FP f6_1(int a, FP b) { return (a += b); }

FP f7(FP a, int b) { return a - b; }
// CIR-LABEL: f7
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!cir.func<!void ()>>>, !cir.ptr<!cir.func<!void ()>>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: %[[SUB:.*]] = cir.unary(minus, %[[STRIDE]]) : !s32i, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!cir.func<!void ()>>, %[[SUB]] : !s32i)

// LLVM-LABEL: f7
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: %[[SUB:.*]] = sub i64 0, %[[STRIDE]]
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[SUB]]

// Similar to f7, just make sure it does not crash.
FP f7_1(FP a, int b) { return (a -= b); }

void f8(void *a, int b) { return *(a + b); }
// CIR-LABEL: f8
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!void>, %[[STRIDE]] : !s32i)
// CIR: cir.return

// LLVM-LABEL: f8
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[STRIDE]]
// LLVM: ret void

void f8_1(void *a, int b) { return a[b]; }
// CIR-LABEL: f8_1
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!void>, %[[STRIDE]] : !s32i)
// CIR: cir.return

// LLVM-LABEL: f8_1
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[STRIDE]]
// LLVM: ret void

unsigned char *p(unsigned int x) {
  unsigned char *p;
  p += 16-x;
  return p;
}

// CIR-LABEL: @p
// CIR: %[[SUB:.*]] = cir.binop(sub
// CIR: cir.ptr_stride({{.*}} : !cir.ptr<!u8i>, %[[SUB]] : !u32i), !cir.ptr<!u8i>

// LLVM-LABEL: @p
// LLVM: getelementptr i8, ptr {{.*}}
