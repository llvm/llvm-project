// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

struct S { unsigned type : 6; unsigned nRefs : 26; };

// Post-increment: returns OLD value, stores OLD+1.
int postinc(S *s) {
  return s->nRefs++;
}

// Pre-increment: returns NEW value, stores OLD+1.
int preinc(S *s) {
  return ++s->nRefs;
}

// CIR-LABEL: @_Z7postincP1S
// CIR:         %[[OLD:.*]] = cir.get_bitfield
// CIR:         %[[INC:.*]] = cir.inc %[[OLD]]
// CIR:         cir.set_bitfield {{.*}} %[[INC]]
// CIR:         cir.cast integral %[[OLD]]
// CIR-NOT:     cir.cast integral %[[INC]]

// CIR-LABEL: @_Z6preincP1S
// CIR:         %[[OLD2:.*]] = cir.get_bitfield
// CIR:         %[[INC2:.*]] = cir.inc %[[OLD2]]
// CIR:         %[[STORED:.*]] = cir.set_bitfield {{.*}} %[[INC2]]
// CIR:         cir.cast integral %[[STORED]]

// Postinc returns the pre-increment value.
// LLVM-LABEL: @_Z7postincP1S
// LLVM:         %[[WORD:.*]] = load i32, ptr %[[PTR:.*]], align 4
// LLVM:         %[[OLD:.*]] = lshr i32 %[[WORD]], 6
// LLVM:         %[[INC:.*]] = add i32 %[[OLD]], 1
// LLVM:         %[[WORD2:.*]] = load i32, ptr %[[PTR]], align 4
// LLVM:         %[[VAL:.*]] = and i32 %[[INC]], 67108863
// LLVM:         %[[SHL:.*]] = shl i32 %[[VAL]], 6
// LLVM:         %[[CLEAR:.*]] = and i32 %[[WORD2]], 63
// LLVM:         %[[SET:.*]] = or i32 %[[CLEAR]], %[[SHL]]
// LLVM:         store i32 %[[SET]], ptr %[[PTR]], align 4
// LLVM-CIR:     store i32 %[[OLD]], ptr %[[RET:.*]], align 4
// LLVM-CIR:     %[[RV:.*]] = load i32, ptr %[[RET]], align 4
// LLVM-CIR:     ret i32 %[[RV]]
// OGCG:         ret i32 %[[OLD]]

// Preinc returns the post-increment (masked new) value.
// LLVM-LABEL: @_Z6preincP1S
// LLVM:         %[[WORD_P:.*]] = load i32, ptr %[[PTR_P:.*]], align 4
// LLVM:         %[[OLD_P:.*]] = lshr i32 %[[WORD_P]], 6
// LLVM:         %[[INC_P:.*]] = add i32 %[[OLD_P]], 1
// LLVM:         %[[WORD2_P:.*]] = load i32, ptr %[[PTR_P]], align 4
// LLVM:         %[[VAL_P:.*]] = and i32 %[[INC_P]], 67108863
// LLVM:         %[[SHL_P:.*]] = shl i32 %[[VAL_P]], 6
// LLVM:         %[[CLEAR_P:.*]] = and i32 %[[WORD2_P]], 63
// LLVM:         %[[SET_P:.*]] = or i32 %[[CLEAR_P]], %[[SHL_P]]
// LLVM:         store i32 %[[SET_P]], ptr %[[PTR_P]], align 4
// LLVM-CIR:     store i32 %[[VAL_P]], ptr %[[RET_P:.*]], align 4
// LLVM-CIR:     %[[RV_P:.*]] = load i32, ptr %[[RET_P]], align 4
// LLVM-CIR:     ret i32 %[[RV_P]]
// OGCG:         ret i32 %[[VAL_P]]
