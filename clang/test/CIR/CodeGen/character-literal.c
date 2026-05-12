// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// A CharacterLiteral '\xFF' has AST type `int` and value 4294967295
// (0xFFFFFFFF; the byte 0xFF reinterpreted as an unsigned 32-bit value).
// Lowering it to an APInt must allow implicit truncation, mirroring
// classic CodeGen's VisitCharacterLiteral.  Without that, constructing
// `APInt(32, 4294967295, /*isSigned=*/true)` would trip the
// `isIntN(...) && "Value is not an N-bit signed value"` assertion in
// LLVM's APInt.h.  Regression test for that path.

void high_byte_to_signed_char(void) {
  signed char c = '\xFF';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @high_byte_to_signed_char
// CIR:         %[[VAL:.*]] = cir.const #cir.int<-1> : !s8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} void @high_byte_to_signed_char
// LLVM:         store i8 -1, ptr %{{.*}}

// OGCG-LABEL: define{{.*}} void @high_byte_to_signed_char
// OGCG:         store i8 -1, ptr %{{.*}}

void boundary_byte(void) {
  signed char c = '\x80';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @boundary_byte
// CIR:         %[[VAL:.*]] = cir.const #cir.int<-128> : !s8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} void @boundary_byte
// LLVM:         store i8 -128, ptr %{{.*}}

// OGCG-LABEL: define{{.*}} void @boundary_byte
// OGCG:         store i8 -128, ptr %{{.*}}

void low_byte(void) {
  signed char c = 'A';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @low_byte
// CIR:         %[[VAL:.*]] = cir.const #cir.int<65> : !s8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} void @low_byte
// LLVM:         store i8 65, ptr %{{.*}}

// OGCG-LABEL: define{{.*}} void @low_byte
// OGCG:         store i8 65, ptr %{{.*}}

void high_byte_to_unsigned_char(void) {
  unsigned char c = '\xFF';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @high_byte_to_unsigned_char
// CIR:         %[[VAL:.*]] = cir.const #cir.int<255> : !u8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-LABEL: define{{.*}} void @high_byte_to_unsigned_char
// LLVM:         store i8 -1, ptr %{{.*}}

// OGCG-LABEL: define{{.*}} void @high_byte_to_unsigned_char
// OGCG:         store i8 -1, ptr %{{.*}}
