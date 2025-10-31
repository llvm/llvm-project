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

// CIR-LABEL: cir.func{{.*}} @_Z2b0ii(
// CIR: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) nsw : !s32i
// CIR: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(rem, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) nsw : !s32i
// CIR: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) nsw : !s32i
// CIR: %{{.+}} = cir.binop(and, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(xor, %{{.+}}, %{{.+}}) : !s32i
// CIR: %{{.+}} = cir.binop(or, %{{.+}}, %{{.+}}) : !s32i
// CIR: cir.return

// LLVM-LABEL: define{{.*}} void @_Z2b0ii(
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

// OGCG-LABEL: define{{.*}} void @_Z2b0ii(i32 {{.*}} %a, i32 {{.*}} %b) {{.*}} { 
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

// CIR-LABEL: cir.func{{.*}} @_Z23testFloatingPointBinOpsff(
// CIR: cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.binop(div, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.binop(add, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.float
// CIR: cir.return

// LLVM-LABEL: define{{.*}} void @_Z23testFloatingPointBinOpsff(
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

// OGCG-LABEL: define{{.*}} void @_Z23testFloatingPointBinOpsff(float {{.*}} %a, float {{.*}} %b)
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

// CIR-LABEL: cir.func{{.*}} @_Z12signed_shiftii(
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

// LLVM-LABEL: define{{.*}} void @_Z12signed_shiftii
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

// OGCG-LABEL: define{{.*}} void @_Z12signed_shiftii
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

// CIR-LABEL: cir.func{{.*}} @_Z14unsigned_shiftjj(
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

// LLVM-LABEL: define{{.*}} void @_Z14unsigned_shiftjj
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

// OGCG-LABEL: define{{.*}} void @_Z14unsigned_shiftjj
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

// CIR-LABEL: cir.func{{.*}} @_Z18zext_shift_exampleih(
// CIR-SAME: %[[ARG0:.*]]: !s32i{{.*}}, %[[ARG1:.*]]: !u8i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !u8i, !cir.ptr<!u8i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[B1_EXT:.*]] = cir.cast integral %[[B1]] : !u8i -> !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s32i, %[[B1_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[B2_EXT:.*]] = cir.cast integral %[[B2]] : !u8i -> !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s32i, %[[B2_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: cir.return

// LLVM-LABEL: define{{.*}} void @_Z18zext_shift_exampleih
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

// OGCG-LABEL: define{{.*}} void @_Z18zext_shift_exampleih
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

// CIR-LABEL: cir.func{{.*}} @_Z18sext_shift_exampleia(
// CIR-SAME: %[[ARG0:.*]]: !s32i{{.*}}, %[[ARG1:.*]]: !s8i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !s8i, !cir.ptr<!s8i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[B1_EXT:.*]] = cir.cast integral %[[B1]] : !s8i -> !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s32i, %[[B1_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[B2_EXT:.*]] = cir.cast integral %[[B2]] : !s8i -> !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s32i, %[[B2_EXT]] : !s32i) -> !s32i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s32i, !cir.ptr<!s32i>

// CIR: cir.return

// LLVM-LABEL: define{{.*}} void @_Z18sext_shift_exampleia
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

// OGCG-LABEL: define{{.*}} void @_Z18sext_shift_exampleia
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

// CIR-LABEL: cir.func{{.*}} @_Z18long_shift_examplexs(
// CIR-SAME: %[[ARG0:.*]]: !s64i{{.*}}, %[[ARG1:.*]]: !s16i{{.*}})
// CIR: %[[A_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["x", init]

// CIR: cir.store{{.*}} %[[ARG0]], %[[A_PTR]] : !s64i, !cir.ptr<!s64i>
// CIR: cir.store{{.*}} %[[ARG1]], %[[B_PTR]] : !s16i, !cir.ptr<!s16i>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: %[[B1_EXT:.*]] = cir.cast integral %[[B1]] : !s16i -> !s32i
// CIR: %[[ASHR:.*]] = cir.shift(right, %[[A1]] : !s64i, %[[B1_EXT]] : !s32i) -> !s64i
// CIR: cir.store{{.*}} %[[ASHR]], %[[X_PTR]] : !s64i, !cir.ptr<!s64i>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: %[[B2_EXT:.*]] = cir.cast integral %[[B2]] : !s16i -> !s32i
// CIR: %[[SHL:.*]] = cir.shift(left, %[[A2]] : !s64i, %[[B2_EXT]] : !s32i) -> !s64i
// CIR: cir.store{{.*}} %[[SHL]], %[[X_PTR]] : !s64i, !cir.ptr<!s64i>

// CIR: cir.return

// LLVM-LABEL: define{{.*}} void @_Z18long_shift_examplexs
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

// OGCG-LABEL: define{{.*}} void @_Z18long_shift_examplexs
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

void b1(bool a, bool b) {
  bool x = a && b;
  x = x || b;
}

// CIR-LABEL: cir.func{{.*}} @_Z2b1bb(
// CIR-SAME: %[[ARG0:.*]]: !cir.bool {{.*}}, %[[ARG1:.*]]: !cir.bool {{.*}})
// CIR: [[A:%[0-9]+]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CIR: [[B:%[0-9]+]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR: [[X:%[0-9]+]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]
// CIR: cir.store %[[ARG0]], [[A]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: cir.store %[[ARG1]], [[B]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: [[AVAL:%[0-9]+]] = cir.load align(1) [[A]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: [[RES1:%[0-9]+]] = cir.ternary([[AVAL]], true {
// CIR: [[BVAL:%[0-9]+]] = cir.load align(1) [[B]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: cir.yield [[BVAL]] : !cir.bool
// CIR: }, false {
// CIR: [[FALSE:%[0-9]+]] = cir.const #false
// CIR: cir.yield [[FALSE]] : !cir.bool
// CIR: }) : (!cir.bool) -> !cir.bool
// CIR: cir.store align(1) [[RES1]], [[X]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: [[XVAL:%[0-9]+]] = cir.load align(1) [[X]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: [[RES2:%[0-9]+]] = cir.ternary([[XVAL]], true {
// CIR: [[TRUE:%[0-9]+]] = cir.const #true
// CIR: cir.yield [[TRUE]] : !cir.bool
// CIR: }, false {
// CIR: [[BVAL2:%[0-9]+]] = cir.load align(1) [[B]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: cir.yield [[BVAL2]] : !cir.bool
// CIR: }) : (!cir.bool) -> !cir.bool
// CIR: cir.store align(1) [[RES2]], [[X]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: cir.return


// LLVM-LABEL: define{{.*}} void @_Z2b1bb(
// LLVM-SAME: i1 %[[ARG0:.+]], i1 %[[ARG1:.+]])
// LLVM: %[[A_ADDR:.*]] = alloca i8
// LLVM: %[[B_ADDR:.*]] = alloca i8
// LLVM: %[[X:.*]] = alloca i8
// LLVM: %[[ZEXT0:.*]] = zext i1 %[[ARG0]] to i8
// LLVM: store i8 %[[ZEXT0]], ptr %[[A_ADDR]]
// LLVM: %[[ZEXT1:.*]] = zext i1 %[[ARG1]] to i8
// LLVM: store i8 %[[ZEXT1]], ptr %[[B_ADDR]]
// LLVM: %[[A_VAL:.*]] = load i8, ptr %[[A_ADDR]]
// LLVM: %[[A_BOOL:.*]] = trunc i8 %[[A_VAL]] to i1
// LLVM: br i1 %[[A_BOOL]], label %[[AND_TRUE:.+]], label %[[AND_FALSE:.+]]
// LLVM: [[AND_TRUE]]:
// LLVM: %[[B_VAL:.*]] = load i8, ptr %[[B_ADDR]]
// LLVM: %[[B_BOOL:.*]] = trunc i8 %[[B_VAL]] to i1
// LLVM: br label %[[AND_MERGE:.+]]
// LLVM: [[AND_FALSE]]:
// LLVM: br label %[[AND_MERGE]]
// LLVM: [[AND_MERGE]]:
// LLVM: %[[AND_PHI:.*]] = phi i1 [ false, %[[AND_FALSE]] ], [ %[[B_BOOL]], %[[AND_TRUE]] ]
// LLVM: %[[ZEXT_AND:.*]] = zext i1 %[[AND_PHI]] to i8
// LLVM: store i8 %[[ZEXT_AND]], ptr %[[X]]
// LLVM: %[[X_VAL:.*]] = load i8, ptr %[[X]]
// LLVM: %[[X_BOOL:.*]] = trunc i8 %[[X_VAL]] to i1
// LLVM: br i1 %[[X_BOOL]], label %[[OR_TRUE:.+]], label %[[OR_FALSE:.+]]
// LLVM: [[OR_TRUE]]:
// LLVM: br label %[[OR_MERGE:.+]]
// LLVM: [[OR_FALSE]]:
// LLVM: %[[B_VAL2:.*]] = load i8, ptr %[[B_ADDR]]
// LLVM: %[[B_BOOL2:.*]] = trunc i8 %[[B_VAL2]] to i1
// LLVM: br label %[[OR_MERGE]]
// LLVM: [[OR_MERGE]]:
// LLVM: %[[OR_PHI:.*]] = phi i1 [ %[[B_BOOL2]], %[[OR_FALSE]] ], [ true, %[[OR_TRUE]] ]
// LLVM: %[[ZEXT_OR:.*]] = zext i1 %[[OR_PHI]] to i8
// LLVM: store i8 %[[ZEXT_OR]], ptr %[[X]]
// LLVM: ret void

// OGCG-LABEL: define{{.*}} void @_Z2b1bb
// OGCG-SAME: (i1 {{.*}} %[[ARG0:.+]], i1 {{.*}} %[[ARG1:.+]])
// OGCG: [[ENTRY:.*]]:
// OGCG: %[[A_ADDR:.*]] = alloca i8
// OGCG: %[[B_ADDR:.*]] = alloca i8
// OGCG: %[[X:.*]] = alloca i8
// OGCG: %[[ZEXT0:.*]] = zext i1 %[[ARG0]] to i8
// OGCG: store i8 %[[ZEXT0]], ptr %[[A_ADDR]]
// OGCG: %[[ZEXT1:.*]] = zext i1 %[[ARG1]] to i8
// OGCG: store i8 %[[ZEXT1]], ptr %[[B_ADDR]]
// OGCG: %[[A_VAL:.*]] = load i8, ptr %[[A_ADDR]]
// OGCG: %[[A_BOOL:.*]] = trunc i8 %[[A_VAL]] to i1
// OGCG: br i1 %[[A_BOOL]], label %[[AND_TRUE:.+]], label %[[AND_MERGE:.+]]
// OGCG: [[AND_TRUE]]:
// OGCG: %[[B_VAL:.*]] = load i8, ptr %[[B_ADDR]]
// OGCG: %[[B_BOOL:.*]] = trunc i8 %[[B_VAL]] to i1
// OGCG: br label %[[AND_MERGE:.+]]
// OGCG: [[AND_MERGE]]:
// OGCG: %[[AND_PHI:.*]] = phi i1 [ false, %[[ENTRY]] ], [ %[[B_BOOL]], %[[AND_TRUE]] ]
// OGCG: %[[ZEXT_AND:.*]] = zext i1 %[[AND_PHI]] to i8
// OGCG: store i8 %[[ZEXT_AND]], ptr %[[X]]
// OGCG: %[[X_VAL:.*]] = load i8, ptr %[[X]]
// OGCG: %[[X_BOOL:.*]] = trunc i8 %[[X_VAL]] to i1
// OGCG: br i1 %[[X_BOOL]], label %[[OR_MERGE:.+]], label %[[OR_FALSE:.+]]
// OGCG: [[OR_FALSE]]:
// OGCG: %[[B_VAL2:.*]] = load i8, ptr %[[B_ADDR]]
// OGCG: %[[B_BOOL2:.*]] = trunc i8 %[[B_VAL2]] to i1
// OGCG: br label %[[OR_MERGE]]
// OGCG: [[OR_MERGE]]:
// OGCG: %[[OR_PHI:.*]] = phi i1 [ true, %[[AND_MERGE]] ], [ %[[B_BOOL2]], %[[OR_FALSE]] ]
// OGCG: %[[ZEXT_OR:.*]] = zext i1 %[[OR_PHI]] to i8
// OGCG: store i8 %[[ZEXT_OR]], ptr %[[X]]
// OGCG: ret void

void b3(int a, int b, int c, int d) {
  bool x = (a == b) && (c == d);
  x = (a == b) || (c == d);
}

// CIR-LABEL: cir.func{{.*}} @_Z2b3iiii(
// CIR-SAME: %[[ARG0:.*]]: !s32i {{.*}}, %[[ARG1:.*]]: !s32i {{.*}}, %[[ARG2:.*]]: !s32i {{.*}}, %[[ARG3:.*]]: !s32i {{.*}})
// CIR: [[A:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: [[B:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: [[C:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR: [[D:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["d", init]
// CIR: [[X:%[0-9]+]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]
// CIR: cir.store %[[ARG0]], [[A]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store %[[ARG1]], [[B]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store %[[ARG2]], [[C]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.store %[[ARG3]], [[D]] : !s32i, !cir.ptr<!s32i>
// CIR: [[AVAL1:%[0-9]+]] = cir.load align(4) [[A]] : !cir.ptr<!s32i>, !s32i
// CIR: [[BVAL1:%[0-9]+]] = cir.load align(4) [[B]] : !cir.ptr<!s32i>, !s32i
// CIR: [[CMP1:%[0-9]+]] = cir.cmp(eq, [[AVAL1]], [[BVAL1]]) : !s32i, !cir.bool
// CIR: [[AND_RESULT:%[0-9]+]] = cir.ternary([[CMP1]], true {
// CIR: [[CVAL1:%[0-9]+]] = cir.load align(4) [[C]] : !cir.ptr<!s32i>, !s32i
// CIR: [[DVAL1:%[0-9]+]] = cir.load align(4) [[D]] : !cir.ptr<!s32i>, !s32i
// CIR: [[CMP2:%[0-9]+]] = cir.cmp(eq, [[CVAL1]], [[DVAL1]]) : !s32i, !cir.bool
// CIR: cir.yield [[CMP2]] : !cir.bool
// CIR: }, false {
// CIR: [[FALSE:%[0-9]+]] = cir.const #false
// CIR: cir.yield [[FALSE]] : !cir.bool
// CIR: }) : (!cir.bool) -> !cir.bool
// CIR: cir.store align(1) [[AND_RESULT]], [[X]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: [[AVAL2:%[0-9]+]] = cir.load align(4) [[A]] : !cir.ptr<!s32i>, !s32i
// CIR: [[BVAL2:%[0-9]+]] = cir.load align(4) [[B]] : !cir.ptr<!s32i>, !s32i
// CIR: [[CMP3:%[0-9]+]] = cir.cmp(eq, [[AVAL2]], [[BVAL2]]) : !s32i, !cir.bool
// CIR: [[OR_RESULT:%[0-9]+]] = cir.ternary([[CMP3]], true {
// CIR: [[TRUE:%[0-9]+]] = cir.const #true
// CIR: cir.yield [[TRUE]] : !cir.bool
// CIR: }, false {
// CIR: [[CVAL2:%[0-9]+]] = cir.load align(4) [[C]] : !cir.ptr<!s32i>, !s32i
// CIR: [[DVAL2:%[0-9]+]] = cir.load align(4) [[D]] : !cir.ptr<!s32i>, !s32i
// CIR: [[CMP4:%[0-9]+]] = cir.cmp(eq, [[CVAL2]], [[DVAL2]]) : !s32i, !cir.bool
// CIR: cir.yield [[CMP4]] : !cir.bool
// CIR: }) : (!cir.bool) -> !cir.bool
// CIR: cir.store align(1) [[OR_RESULT]], [[X]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: cir.return


// LLVM-LABEL: define{{.*}} void @_Z2b3iiii(
// LLVM-SAME: i32 %[[ARG0:.+]], i32 %[[ARG1:.+]], i32 %[[ARG2:.+]], i32 %[[ARG3:.+]])
// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1
// LLVM: %[[C_ADDR:.*]] = alloca i32, i64 1
// LLVM: %[[D_ADDR:.*]] = alloca i32, i64 1
// LLVM: %[[X:.*]] = alloca i8, i64 1
// LLVM: store i32 %[[ARG0]], ptr %[[A_ADDR]]
// LLVM: store i32 %[[ARG1]], ptr %[[B_ADDR]]
// LLVM: store i32 %[[ARG2]], ptr %[[C_ADDR]]
// LLVM: store i32 %[[ARG3]], ptr %[[D_ADDR]]
// LLVM: %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM: %[[B_VAL:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM: %[[CMP1:.*]] = icmp eq i32 %[[A_VAL]], %[[B_VAL]]
// LLVM: br i1 %[[CMP1]], label %[[AND_TRUE:.+]], label %[[AND_FALSE:.+]]
// LLVM: [[AND_TRUE]]:
// LLVM: %[[C_VAL:.*]] = load i32, ptr %[[C_ADDR]]
// LLVM: %[[D_VAL:.*]] = load i32, ptr %[[D_ADDR]]
// LLVM: %[[CMP2:.*]] = icmp eq i32 %[[C_VAL]], %[[D_VAL]]
// LLVM: br label %[[AND_MERGE:.+]]
// LLVM: [[AND_FALSE]]:
// LLVM: br label %[[AND_MERGE]]
// LLVM: [[AND_MERGE]]:
// LLVM: %[[AND_PHI:.*]] = phi i1 [ false, %[[AND_FALSE]] ], [ %[[CMP2]], %[[AND_TRUE]] ]
// LLVM: %[[ZEXT_AND:.*]] = zext i1 %[[AND_PHI]] to i8
// LLVM: store i8 %[[ZEXT_AND]], ptr %[[X]]
// LLVM: %[[A_VAL2:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM: %[[B_VAL2:.*]] = load i32, ptr %[[B_ADDR]]
// LLVM: %[[CMP3:.*]] = icmp eq i32 %[[A_VAL2]], %[[B_VAL2]]
// LLVM: br i1 %[[CMP3]], label %[[OR_TRUE:.+]], label %[[OR_FALSE:.+]]
// LLVM: [[OR_TRUE]]:
// LLVM: br label %[[OR_MERGE:.+]]
// LLVM: [[OR_FALSE]]:
// LLVM: %[[C_VAL2:.*]] = load i32, ptr %[[C_ADDR]]
// LLVM: %[[D_VAL2:.*]] = load i32, ptr %[[D_ADDR]]
// LLVM: %[[CMP4:.*]] = icmp eq i32 %[[C_VAL2]], %[[D_VAL2]]
// LLVM: br label %[[OR_MERGE]]
// LLVM: [[OR_MERGE]]:
// LLVM: %[[OR_PHI:.*]] = phi i1 [ %[[CMP4]], %[[OR_FALSE]] ], [ true, %[[OR_TRUE]] ]
// LLVM: %[[ZEXT_OR:.*]] = zext i1 %[[OR_PHI]] to i8
// LLVM: store i8 %[[ZEXT_OR]], ptr %[[X]]
// LLVM: ret void

// OGCG-LABEL: define{{.*}} void @_Z2b3iiii(
// OGCG-SAME: i32 {{.*}} %[[ARG0:.+]], i32 {{.*}} %[[ARG1:.+]], i32 {{.*}} %[[ARG2:.+]], i32 {{.*}} %[[ARG3:.+]])
// OGCG: [[ENTRY:.*]]:
// OGCG: %[[A_ADDR:.*]] = alloca i32
// OGCG: %[[B_ADDR:.*]] = alloca i32
// OGCG: %[[C_ADDR:.*]] = alloca i32
// OGCG: %[[D_ADDR:.*]] = alloca i32
// OGCG: %[[X:.*]] = alloca i8
// OGCG: store i32 %[[ARG0]], ptr %[[A_ADDR]]
// OGCG: store i32 %[[ARG1]], ptr %[[B_ADDR]]
// OGCG: store i32 %[[ARG2]], ptr %[[C_ADDR]]
// OGCG: store i32 %[[ARG3]], ptr %[[D_ADDR]]
// OGCG: %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG: %[[B_VAL:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG: %[[CMP1:.*]] = icmp eq i32 %[[A_VAL]], %[[B_VAL]]
// OGCG: br i1 %[[CMP1]], label %[[AND_TRUE:.+]], label %[[AND_MERGE:.+]]
// OGCG: [[AND_TRUE]]:
// OGCG: %[[C_VAL:.*]] = load i32, ptr %[[C_ADDR]]
// OGCG: %[[D_VAL:.*]] = load i32, ptr %[[D_ADDR]]
// OGCG: %[[CMP2:.*]] = icmp eq i32 %[[C_VAL]], %[[D_VAL]]
// OGCG: br label %[[AND_MERGE:.+]]
// OGCG: [[AND_MERGE]]:
// OGCG: %[[AND_PHI:.*]] = phi i1 [ false, %[[ENTRY]] ], [ %[[CMP2]], %[[AND_TRUE]] ]
// OGCG: %[[ZEXT_AND:.*]] = zext i1 %[[AND_PHI]] to i8
// OGCG: store i8 %[[ZEXT_AND]], ptr %[[X]]
// OGCG: %[[A_VAL2:.*]] = load i32, ptr %[[A_ADDR]]
// OGCG: %[[B_VAL2:.*]] = load i32, ptr %[[B_ADDR]]
// OGCG: %[[CMP3:.*]] = icmp eq i32 %[[A_VAL2]], %[[B_VAL2]]
// OGCG: br i1 %[[CMP3]], label %[[OR_MERGE:.+]], label %[[OR_FALSE:.+]]
// OGCG: [[OR_FALSE]]:
// OGCG: %[[C_VAL2:.*]] = load i32, ptr %[[C_ADDR]]
// OGCG: %[[D_VAL2:.*]] = load i32, ptr %[[D_ADDR]]
// OGCG: %[[CMP4:.*]] = icmp eq i32 %[[C_VAL2]], %[[D_VAL2]]
// OGCG: br label %[[OR_MERGE]]
// OGCG: [[OR_MERGE]]:
// OGCG: %[[OR_PHI:.*]] = phi i1 [ true, %[[AND_MERGE]] ], [ %[[CMP4]], %[[OR_FALSE]] ]
// OGCG: %[[ZEXT_OR:.*]] = zext i1 %[[OR_PHI]] to i8
// OGCG: store i8 %[[ZEXT_OR]], ptr %[[X]]
// OGCG: ret void
