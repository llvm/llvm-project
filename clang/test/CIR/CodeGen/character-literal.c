// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void high_byte_to_signed_char(void) {
  signed char c = '\xFF';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @high_byte_to_signed_char
// CIR:         %[[VAL:.*]] = cir.const #cir.int<-1> : !s8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} void @high_byte_to_signed_char
// LLVM:         store i8 -1, ptr %{{.*}}

void boundary_byte(void) {
  signed char c = '\x80';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @boundary_byte
// CIR:         %[[VAL:.*]] = cir.const #cir.int<-128> : !s8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} void @boundary_byte
// LLVM:         store i8 -128, ptr %{{.*}}

void low_byte(void) {
  signed char c = 'A';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @low_byte
// CIR:         %[[VAL:.*]] = cir.const #cir.int<65> : !s8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} void @low_byte
// LLVM:         store i8 65, ptr %{{.*}}

void high_byte_to_unsigned_char(void) {
  unsigned char c = '\xFF';
  (void)c;
}

// CIR-LABEL: cir.func{{.*}} @high_byte_to_unsigned_char
// CIR:         %[[VAL:.*]] = cir.const #cir.int<255> : !u8i
// CIR:         cir.store{{.*}} %[[VAL]], %{{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-LABEL: define{{.*}} void @high_byte_to_unsigned_char
// LLVM:         store i8 -1, ptr %{{.*}}
