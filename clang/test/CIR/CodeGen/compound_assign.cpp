// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int compound_assign(int b) {
  int x = 1;
  x *= b;
  x /= b;
  x %= b;
  x += b;
  x -= b;
  x >>= b;
  x <<= b;
  x &= b;
  x ^= b;
  x |= b;
  return x;
}

// CIR: cir.func @_Z15compound_assigni
// CIR:   %[[MUL:.*]] = cir.binop(mul, %{{.*}}, %{{.*}}) nsw : !s32i
// CIR:   cir.store{{.*}} %[[MUL]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[DIV:.*]] = cir.binop(div, %{{.*}}, %{{.*}}) : !s32i
// CIR:   cir.store{{.*}} %[[DIV]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[REM:.*]] = cir.binop(rem, %{{.*}}, %{{.*}}) : !s32i
// CIR:   cir.store{{.*}} %[[REM]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ADD:.*]] = cir.binop(add, %{{.*}}, %{{.*}}) nsw : !s32i
// CIR:   cir.store{{.*}} %[[ADD]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[SUB:.*]] = cir.binop(sub, %{{.*}}, %{{.*}}) nsw : !s32i
// CIR:   cir.store{{.*}} %[[SUB]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[SHR:.*]] = cir.shift(right, %{{.*}} : !s32i, %{{.*}} : !s32i) -> !s32i
// CIR:   cir.store{{.*}} %[[SHR]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[SHL:.*]] = cir.shift(left, %{{.*}} : !s32i, %{{.*}} : !s32i) -> !s32i
// CIR:   cir.store{{.*}} %[[SHL]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[AND:.*]] = cir.binop(and, %{{.*}}, %{{.*}}) : !s32i
// CIR:   cir.store{{.*}} %[[AND]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[XOR:.*]] = cir.binop(xor, %{{.*}}, %{{.*}}) : !s32i
// CIR:   cir.store{{.*}} %[[XOR]], %{{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   %[[OR:.*]] = cir.binop(or, %{{.*}}, %{{.*}}) : !s32i
// CIR:   cir.store{{.*}} %[[OR]], %{{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM: define {{.*}}i32 @_Z15compound_assigni
// LLVM:   %[[MUL:.*]] = mul nsw i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[MUL]], ptr %{{.*}}
// LLVM:   %[[DIV:.*]] = sdiv i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[DIV]], ptr %{{.*}}
// LLVM:   %[[REM:.*]] = srem i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[REM]], ptr %{{.*}}
// LLVM:   %[[ADD:.*]] = add nsw i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[ADD]], ptr %{{.*}}
// LLVM:   %[[SUB:.*]] = sub nsw i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[SUB]], ptr %{{.*}}
// LLVM:   %[[SHR:.*]] = ashr i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[SHR]], ptr %{{.*}}
// LLVM:   %[[SHL:.*]] = shl i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[SHL]], ptr %{{.*}}
// LLVM:   %[[AND:.*]] = and i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[AND]], ptr %{{.*}}
// LLVM:   %[[XOR:.*]] = xor i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[XOR]], ptr %{{.*}}
// LLVM:   %[[OR:.*]] = or i32 %{{.*}}, %{{.*}}
// LLVM:   store i32 %[[OR]], ptr %{{.*}}

// OGCG: define {{.*}}i32 @_Z15compound_assigni
// OGCG:   %[[MUL:.*]] = mul nsw i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[MUL]], ptr %{{.*}}
// OGCG:   %[[DIV:.*]] = sdiv i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[DIV]], ptr %{{.*}}
// OGCG:   %[[REM:.*]] = srem i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[REM]], ptr %{{.*}}
// OGCG:   %[[ADD:.*]] = add nsw i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[ADD]], ptr %{{.*}}
// OGCG:   %[[SUB:.*]] = sub nsw i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[SUB]], ptr %{{.*}}
// OGCG:   %[[SHR:.*]] = ashr i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[SHR]], ptr %{{.*}}
// OGCG:   %[[SHL:.*]] = shl i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[SHL]], ptr %{{.*}}
// OGCG:   %[[AND:.*]] = and i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[AND]], ptr %{{.*}}
// OGCG:   %[[XOR:.*]] = xor i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[XOR]], ptr %{{.*}}
// OGCG:   %[[OR:.*]] = or i32 %{{.*}}, %{{.*}}
// OGCG:   store i32 %[[OR]], ptr %{{.*}}
