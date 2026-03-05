// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

unsigned char bitreverse8(unsigned char value) {
  return __builtin_bitreverse8(value);
}

// CIR-LABEL: @bitreverse8
// CIR: %{{.+}} = cir.bit_reverse %{{.+}} : !u8i

// LLVM-LABEL: @bitreverse8
// LLVM: %{{.+}} = call i8 @llvm.bitreverse.i8(i8 %{{.+}})

unsigned short bitreverse16(unsigned short value) {
  return __builtin_bitreverse16(value);
}

// CIR-LABEL: @bitreverse16
// CIR: %{{.+}} = cir.bit_reverse %{{.+}} : !u16i

// LLVM-LABEL: @bitreverse16
// LLVM: %{{.+}} = call i16 @llvm.bitreverse.i16(i16 %{{.+}})

unsigned bitreverse32(unsigned value) {
  return __builtin_bitreverse32(value);
}

// CIR-LABEL: @bitreverse32
// CIR: %{{.+}} = cir.bit_reverse %{{.+}} : !u32i

// LLVM-LABEL: @bitreverse32
// LLVM: %{{.+}} = call i32 @llvm.bitreverse.i32(i32 %{{.+}})

unsigned long long bitreverse64(unsigned long long value) {
  return __builtin_bitreverse64(value);
}

// CIR-LABEL: @bitreverse64
// CIR: %{{.+}} = cir.bit_reverse %{{.+}} : !u64i

// LLVM-LABEL: @bitreverse64
// LLVM: %{{.+}} = call i64 @llvm.bitreverse.i64(i64 %{{.+}})
