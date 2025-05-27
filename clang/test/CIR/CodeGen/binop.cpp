// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void b0(int a, int b) {
  int x = a * b;
  x = x / b;
  x = x % b;
  x = x + b;
  x = x - b;
  x = x & b;
  x = x ^ b;
  x = x | b;
}

// CIR-LABEL: cir.func @_Z2b0ii(
// CIR: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) nsw : !s32i
// CIR: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(rem, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) nsw : !s32i
// CIR: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) nsw : !s32i
// CIR: %{{.+}} = cir.binop(and, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(xor, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(or, %{{.+}}, %{{.+}}) : !s32i
// CIR: cir.return

// LLVM-LABEL: define void @_Z2b0ii(
// LLVM-SAME: i32 %[[A:.*]], i32 %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca i32
// LLVM:         %[[B_ADDR:.*]] = alloca i32
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         store i32 %[[A]], ptr %[[A_ADDR]]
// LLVM:         store i32 %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[MUL:.*]] = mul nsw i32 %[[A]], %[[B]]
// LLVM:         store i32 %[[MUL]], ptr %[[X]]

// LLVM:         %[[X1:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B1:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[DIV:.*]] = sdiv i32 %[[X1]], %[[B1]]
// LLVM:         store i32 %[[DIV]], ptr %[[X]]

// LLVM:         %[[X2:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B2:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[REM:.*]] = srem i32 %[[X2]], %[[B2]]
// LLVM:         store i32 %[[REM]], ptr %[[X]]

// LLVM:         %[[X3:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B3:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[ADD:.*]] = add nsw i32 %[[X3]], %[[B3]]
// LLVM:         store i32 %[[ADD]], ptr %[[X]]

// LLVM:         %[[X4:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B4:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[SUB:.*]] = sub nsw i32 %[[X4]], %[[B4]]
// LLVM:         store i32 %[[SUB]], ptr %[[X]]

// LLVM:         %[[X5:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B5:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[AND:.*]] = and i32 %[[X5]], %[[B5]]
// LLVM:         store i32 %[[AND]], ptr %[[X]]

// LLVM:         %[[X6:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B6:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[XOR:.*]] = xor i32 %[[X6]], %[[B6]]
// LLVM:         store i32 %[[XOR]], ptr %[[X]]

// LLVM:         %[[X7:.*]] = load i32, ptr %[[X]]
// LLVM:         %[[B7:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[OR:.*]] = or i32 %[[X7]], %[[B7]]
// LLVM:         store i32 %[[OR]], ptr %[[X]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z2b0ii(i32 {{.*}} %a, i32 {{.*}} %b) {{.*}} { 
// OGCG:         %[[A_ADDR:.*]] = alloca i32
// OGCG:         %[[B_ADDR:.*]] = alloca i32
// OGCG:         %[[X:.*]] = alloca i32
// OGCG:         store i32 %a, ptr %[[A_ADDR]]
// OGCG:         store i32 %b, ptr %[[B_ADDR]]

// OGCG:         %[[A:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[MUL:.*]] = mul nsw i32 %[[A]], %[[B]]
// OGCG:         store i32 %[[MUL]], ptr %[[X]]

// OGCG:         %[[X1:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B1:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[DIV:.*]] = sdiv i32 %[[X1]], %[[B1]]
// OGCG:         store i32 %[[DIV]], ptr %[[X]]

// OGCG:         %[[X2:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B2:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[REM:.*]] = srem i32 %[[X2]], %[[B2]]
// OGCG:         store i32 %[[REM]], ptr %[[X]]

// OGCG:         %[[X3:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B3:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[ADD:.*]] = add nsw i32 %[[X3]], %[[B3]]
// OGCG:         store i32 %[[ADD]], ptr %[[X]]

// OGCG:         %[[X4:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B4:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[SUB:.*]] = sub nsw i32 %[[X4]], %[[B4]]
// OGCG:         store i32 %[[SUB]], ptr %[[X]]

// OGCG:         %[[X5:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B5:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[AND:.*]] = and i32 %[[X5]], %[[B5]]
// OGCG:         store i32 %[[AND]], ptr %[[X]]

// OGCG:         %[[X6:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B6:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[XOR:.*]] = xor i32 %[[X6]], %[[B6]]
// OGCG:         store i32 %[[XOR]], ptr %[[X]]

// OGCG:         %[[X7:.*]] = load i32, ptr %[[X]]
// OGCG:         %[[B7:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[OR:.*]] = or i32 %[[X7]], %[[B7]]
// OGCG:         store i32 %[[OR]], ptr %[[X]]

// OGCG:         ret void

void testFloatingPointBinOps(float a, float b) {
  a * b;
  a / b;
  a + b;
  a - b;
}

// CIR-LABEL: cir.func @_Z23testFloatingPointBinOpsff(
// CIR: cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.binop(div, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.binop(add, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.return

// LLVM-LABEL: define void @_Z23testFloatingPointBinOpsff(
// LLVM-SAME: float %[[A:.*]], float %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca float, i64 1
// LLVM:         %[[B_ADDR:.*]] = alloca float, i64 1
// LLVM:         store float %[[A]], ptr %[[A_ADDR]]
// LLVM:         store float %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A1:.*]] = load float, ptr %[[A_ADDR]]
// LLVM:         %[[B1:.*]] = load float, ptr %[[B_ADDR]]
// LLVM:         fmul float %[[A1]], %[[B1]]

// LLVM:         %[[A2:.*]] = load float, ptr %[[A_ADDR]]
// LLVM:         %[[B2:.*]] = load float, ptr %[[B_ADDR]]
// LLVM:         fdiv float %[[A2]], %[[B2]]

// LLVM:         %[[A3:.*]] = load float, ptr %[[A_ADDR]]
// LLVM:         %[[B3:.*]] = load float, ptr %[[B_ADDR]]
// LLVM:         fadd float %[[A3]], %[[B3]]

// LLVM:         %[[A4:.*]] = load float, ptr %[[A_ADDR]]
// LLVM:         %[[B4:.*]] = load float, ptr %[[B_ADDR]]
// LLVM:         fsub float %[[A4]], %[[B4]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z23testFloatingPointBinOpsff(float {{.*}} %a, float {{.*}} %b)
// OGCG:         %a.addr = alloca float
// OGCG:         %b.addr = alloca float
// OGCG:         store float %a, ptr %a.addr
// OGCG:         store float %b, ptr %b.addr

// OGCG:         %[[A1:.*]] = load float, ptr %a.addr
// OGCG:         %[[B1:.*]] = load float, ptr %b.addr
// OGCG:         fmul float %[[A1]], %[[B1]]

// OGCG:         %[[A2:.*]] = load float, ptr %a.addr
// OGCG:         %[[B2:.*]] = load float, ptr %b.addr
// OGCG:         fdiv float %[[A2]], %[[B2]]

// OGCG:         %[[A3:.*]] = load float, ptr %a.addr
// OGCG:         %[[B3:.*]] = load float, ptr %b.addr
// OGCG:         fadd float %[[A3]], %[[B3]]

// OGCG:         %[[A4:.*]] = load float, ptr %a.addr
// OGCG:         %[[B4:.*]] = load float, ptr %b.addr
// OGCG:         fsub float %[[A4]], %[[B4]]

// OGCG:         ret void

void signed_shift(int a, int b) {
  int x = a >> b;
  x = a << b;
}

// CIR-LABEL: cir.func @_Z12signed_shiftii(
// CIR-SAME: %[[ARG0:.*]]: !s32i{{.*}}, %[[ARG1:.*]]: !s32i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s32i, %[[B1]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s32i, %[[B2]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: cir.return

// LLVM-LABEL: define void @_Z12signed_shiftii
// LLVM-SAME: (i32 %[[A:.*]], i32 %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca i32
// LLVM:         %[[B_ADDR:.*]] = alloca i32
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         store i32 %[[A]], ptr %[[A_ADDR]]
// LLVM:         store i32 %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B1:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[ASHR:.*]] = ashr i32 %[[A1]], %[[B1]]
// LLVM:         store i32 %[[ASHR]], ptr %[[X]]

// LLVM:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B2:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2]]
// LLVM:         store i32 %[[SHL]], ptr %[[X]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z12signed_shiftii
// OGCG-SAME: (i32 {{.*}} %[[A:.*]], i32 {{.*}} %[[B:.*]])
// OGCG:         %[[A_ADDR:.*]] = alloca i32
// OGCG:         %[[B_ADDR:.*]] = alloca i32
// OGCG:         %[[X:.*]] = alloca i32
// OGCG:         store i32 %[[A]], ptr %[[A_ADDR]]
// OGCG:         store i32 %[[B]], ptr %[[B_ADDR]]

// OGCG:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B1:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[ASHR:.*]] = ashr i32 %[[A1]], %[[B1]]
// OGCG:         store i32 %[[ASHR]], ptr %[[X]]

// OGCG:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B2:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2]]
// OGCG:         store i32 %[[SHL]], ptr %[[X]]

// OGCG:         ret void

void unsigned_shift(unsigned a, unsigned b) {
  unsigned x = a >> b;
  x = a << b;
}

// CIR-LABEL: cir.func @_Z14unsigned_shiftjj(
// CIR-SAME: %[[ARG0:.*]]: !u32i{{.*}}, %[[ARG1:.*]]: !u32i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !u32i, !cir.ptr<!u32i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !u32i, !cir.ptr<!u32i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !u32i, %[[B1]] : !u32i) -> !u32i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !u32i, !cir.ptr<!u32i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !u32i, %[[B2]] : !u32i) -> !u32i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !u32i, !cir.ptr<!u32i>

// CIR: cir.return

// LLVM-LABEL: define void @_Z14unsigned_shiftjj
// LLVM-SAME: (i32 %[[A:.*]], i32 %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca i32
// LLVM:         %[[B_ADDR:.*]] = alloca i32
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         store i32 %[[A]], ptr %[[A_ADDR]]
// LLVM:         store i32 %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B1:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[ASHR:.*]] = lshr i32 %[[A1]], %[[B1]]
// LLVM:         store i32 %[[ASHR]], ptr %[[X]]

// LLVM:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B2:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2]]
// LLVM:         store i32 %[[SHL]], ptr %[[X]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z14unsigned_shiftjj
// OGCG-SAME: (i32 {{.*}} %[[A:.*]], i32 {{.*}} %[[B:.*]])
// OGCG:         %[[A_ADDR:.*]] = alloca i32
// OGCG:         %[[B_ADDR:.*]] = alloca i32
// OGCG:         %[[X:.*]] = alloca i32
// OGCG:         store i32 %[[A]], ptr %[[A_ADDR]]
// OGCG:         store i32 %[[B]], ptr %[[B_ADDR]]

// OGCG:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B1:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[ASHR:.*]] = lshr i32 %[[A1]], %[[B1]]
// OGCG:         store i32 %[[ASHR]], ptr %[[X]]

// OGCG:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B2:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2]]
// OGCG:         store i32 %[[SHL]], ptr %[[X]]

// OGCG:         ret void

void zext_shift_example(int a, unsigned char b) {
  int x = a >> b;
  x = a << b;
}

// CIR-LABEL: cir.func @_Z18zext_shift_exampleih(
// CIR-SAME: %[[ARG0:.*]]: !s32i{{.*}}, %[[ARG1:.*]]: !u8i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !u8i, !cir.ptr<!u8i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[B1_EXT:.*]] = cir.cast(integral, %[[B1]] : !u8i), !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s32i, %[[B1_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[B2_EXT:.*]] = cir.cast(integral, %[[B2]] : !u8i), !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s32i, %[[B2_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: cir.return

// LLVM-LABEL: define void @_Z18zext_shift_exampleih
// LLVM-SAME: (i32 %[[A:.*]], i8 %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca i32
// LLVM:         %[[B_ADDR:.*]] = alloca i8
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         store i32 %[[A]], ptr %[[A_ADDR]]
// LLVM:         store i8 %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B1:.*]] = load i8, ptr %[[B_ADDR]]
// LLVM:         %[[B1_EXT:.*]] = zext i8 %[[B1]] to i32
// LLVM:         %[[ASHR:.*]] = ashr i32 %[[A1]], %[[B1_EXT]]
// LLVM:         store i32 %[[ASHR]], ptr %[[X]]

// LLVM:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B2:.*]] = load i8, ptr %[[B_ADDR]]
// LLVM:         %[[B2_EXT:.*]] = zext i8 %[[B2]] to i32
// LLVM:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2_EXT]]
// LLVM:         store i32 %[[SHL]], ptr %[[X]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z18zext_shift_exampleih
// OGCG-SAME: (i32 {{.*}} %[[A:.*]], i8 {{.*}} %[[B:.*]])
// OGCG:         %[[A_ADDR:.*]] = alloca i32
// OGCG:         %[[B_ADDR:.*]] = alloca i8
// OGCG:         %[[X:.*]] = alloca i32
// OGCG:         store i32 %[[A]], ptr %[[A_ADDR]]
// OGCG:         store i8 %[[B]], ptr %[[B_ADDR]]

// OGCG:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B1:.*]] = load i8, ptr %[[B_ADDR]]
// OGCG:         %[[B1_EXT:.*]] = zext i8 %[[B1]] to i32
// OGCG:         %[[ASHR:.*]] = ashr i32 %[[A1]], %[[B1_EXT]]
// OGCG:         store i32 %[[ASHR]], ptr %[[X]]

// OGCG:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B2:.*]] = load i8, ptr %[[B_ADDR]]
// OGCG:         %[[B2_EXT:.*]] = zext i8 %[[B2]] to i32
// OGCG:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2_EXT]]
// OGCG:         store i32 %[[SHL]], ptr %[[X]]

// OGCG:         ret void

void sext_shift_example(int a, signed char b) {
  int x = a >> b;
  x = a << b;
}

// CIR-LABEL: cir.func @_Z18sext_shift_exampleia(
// CIR-SAME: %[[ARG0:.*]]: !s32i{{.*}}, %[[ARG1:.*]]: !s8i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !s8i, !cir.ptr<!s8i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[B1_EXT:.*]] = cir.cast(integral, %[[B1]] : !s8i), !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s32i, %[[B1_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[B2_EXT:.*]] = cir.cast(integral, %[[B2]] : !s8i), !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s32i, %[[B2_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: cir.return

// LLVM-LABEL: define void @_Z18sext_shift_exampleia
// LLVM-SAME: (i32 %[[A:.*]], i8 %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca i32
// LLVM:         %[[B_ADDR:.*]] = alloca i8
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         store i32 %[[A]], ptr %[[A_ADDR]]
// LLVM:         store i8 %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B1:.*]] = load i8, ptr %[[B_ADDR]]
// LLVM:         %[[B1_EXT:.*]] = sext i8 %[[B1]] to i32
// LLVM:         %[[ASHR:.*]] = ashr i32 %[[A1]], %[[B1_EXT]]
// LLVM:         store i32 %[[ASHR]], ptr %[[X]]

// LLVM:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM:         %[[B2:.*]] = load i8, ptr %[[B_ADDR]]
// LLVM:         %[[B2_EXT:.*]] = sext i8 %[[B2]] to i32
// LLVM:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2_EXT]]
// LLVM:         store i32 %[[SHL]], ptr %[[X]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z18sext_shift_exampleia
// OGCG-SAME: (i32 {{.*}} %[[A:.*]], i8 {{.*}} %[[B:.*]])
// OGCG:         %[[A_ADDR:.*]] = alloca i32
// OGCG:         %[[B_ADDR:.*]] = alloca i8
// OGCG:         %[[X:.*]] = alloca i32
// OGCG:         store i32 %[[A]], ptr %[[A_ADDR]]
// OGCG:         store i8 %[[B]], ptr %[[B_ADDR]]

// OGCG:         %[[A1:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B1:.*]] = load i8, ptr %[[B_ADDR]]
// OGCG:         %[[B1_EXT:.*]] = sext i8 %[[B1]] to i32
// OGCG:         %[[ASHR:.*]] = ashr i32 %[[A1]], %[[B1_EXT]]
// OGCG:         store i32 %[[ASHR]], ptr %[[X]]

// OGCG:         %[[A2:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG:         %[[B2:.*]] = load i8, ptr %[[B_ADDR]]
// OGCG:         %[[B2_EXT:.*]] = sext i8 %[[B2]] to i32
// OGCG:         %[[SHL:.*]] = shl i32 %[[A2]], %[[B2_EXT]]
// OGCG:         store i32 %[[SHL]], ptr %[[X]]

// OGCG:         ret void

void long_shift_example(long long a, short b) {
  long long x = a >> b;
  x = a << b;
}

// CIR-LABEL: cir.func @_Z18long_shift_examplexs(
// CIR-SAME: %[[ARG0:.*]]: !s64i{{.*}}, %[[ARG1:.*]]: !s16i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s64i, !cir.ptr<!s64i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !s16i, !cir.ptr<!s16i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: %[[B1_EXT:.*]] = cir.cast(integral, %[[B1]] : !s16i), !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s64i, %[[B1_EXT]] : !s32i) -> !s64i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s64i, !cir.ptr<!s64i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: %[[B2_EXT:.*]] = cir.cast(integral, %[[B2]] : !s16i), !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s64i, %[[B2_EXT]] : !s32i) -> !s64i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s64i, !cir.ptr<!s64i>

// CIR: cir.return

// LLVM-LABEL: define void @_Z18long_shift_examplexs
// LLVM-SAME: (i64 %[[A:.*]], i16 %[[B:.*]])
// LLVM:         %[[A_ADDR:.*]] = alloca i64
// LLVM:         %[[B_ADDR:.*]] = alloca i16
// LLVM:         %[[X:.*]] = alloca i64
// LLVM:         store i64 %[[A]], ptr %[[A_ADDR]]
// LLVM:         store i16 %[[B]], ptr %[[B_ADDR]]

// LLVM:         %[[A1:.*]] = load i64, ptr %[[A_ADDR]]
// LLVM:         %[[B1:.*]] = load i16, ptr %[[B_ADDR]]
// LLVM:         %[[B1_SEXT:.*]] = sext i16 %[[B1]] to i32
// LLVM:         %[[B1_ZEXT:.*]] = zext i32 %[[B1_SEXT]] to i64
// LLVM:         %[[ASHR:.*]] = ashr i64 %[[A1]], %[[B1_ZEXT]]
// LLVM:         store i64 %[[ASHR]], ptr %[[X]]

// LLVM:         %[[A2:.*]] = load i64, ptr %[[A_ADDR]]
// LLVM:         %[[B2:.*]] = load i16, ptr %[[B_ADDR]]
// LLVM:         %[[B2_SEXT:.*]] = sext i16 %[[B2]] to i32
// LLVM:         %[[B2_ZEXT:.*]] = zext i32 %[[B2_SEXT]] to i64
// LLVM:         %[[SHL:.*]] = shl i64 %[[A2]], %[[B2_ZEXT]]
// LLVM:         store i64 %[[SHL]], ptr %[[X]]

// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z18long_shift_examplexs
// OGCG-SAME: (i64 {{.*}} %[[A:.*]], i16 {{.*}} %[[B:.*]])
// OGCG:         %[[A_ADDR:.*]] = alloca i64
// OGCG:         %[[B_ADDR:.*]] = alloca i16
// OGCG:         %[[X:.*]] = alloca i64
// OGCG:         store i64 %[[A]], ptr %[[A_ADDR]]
// OGCG:         store i16 %[[B]], ptr %[[B_ADDR]]

// OGCG:         %[[A1:.*]] = load i64, ptr %[[A_ADDR]]
// OGCG:         %[[B1:.*]] = load i16, ptr %[[B_ADDR]]
// OGCG:         %[[B1_SEXT:.*]] = sext i16 %[[B1]] to i32
// OGCG:         %[[B1_ZEXT:.*]] = zext i32 %[[B1_SEXT]] to i64
// OGCG:         %[[ASHR:.*]] = ashr i64 %[[A1]], %[[B1_ZEXT]]
// OGCG:         store i64 %[[ASHR]], ptr %[[X]]

// OGCG:         %[[A2:.*]] = load i64, ptr %[[A_ADDR]]
// OGCG:         %[[B2:.*]] = load i16, ptr %[[B_ADDR]]
// OGCG:         %[[B2_SEXT:.*]] = sext i16 %[[B2]] to i32
// OGCG:         %[[B2_ZEXT:.*]] = zext i32 %[[B2_SEXT]] to i64
// OGCG:         %[[SHL:.*]] = shl i64 %[[A2]], %[[B2_ZEXT]]
// OGCG:         store i64 %[[SHL]], ptr %[[X]]

// OGCG:         ret void
