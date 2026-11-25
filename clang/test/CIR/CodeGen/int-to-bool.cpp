// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

bool f1(unsigned char c) {
  return c;
}

// CIR: cir.func{{.*}} @_Z2f1h
// CIR:   cir.cast int_to_bool %{{.*}} : !u8i -> !cir.bool

// LLVM: define{{.*}} i1 @_Z2f1h
// LLVM:   %[[CMP:.*]] = icmp ne i8 %{{.*}}, 0
// LLVM:   ret i1 %[[CMP]]

bool f2(short s) {
  return s;
}

// CIR: cir.func{{.*}} @_Z2f2s
// CIR:   cir.cast int_to_bool %{{.*}} : !s16i -> !cir.bool

// LLVM: define{{.*}} i1 @_Z2f2s
// LLVM:   %[[CMP:.*]] = icmp ne i16 %{{.*}}, 0
// LLVM:   ret i1 %[[CMP]]

bool f3(unsigned u) {
  return u;
}

// CIR: cir.func{{.*}} @_Z2f3j
// CIR:   cir.cast int_to_bool %{{.*}} : !u32i -> !cir.bool

// LLVM: define{{.*}} i1 @_Z2f3j
// LLVM:   %[[CMP:.*]] = icmp ne i32 %{{.*}}, 0
// LLVM:   ret i1 %[[CMP]]

bool f4(long l) {
  return l;
}

// CIR: cir.func{{.*}} @_Z2f4l
// CIR:   cir.cast int_to_bool %{{.*}} : !s64i -> !cir.bool

// LLVM: define{{.*}} i1 @_Z2f4l
// LLVM:   %[[CMP:.*]] = icmp ne i64 %{{.*}}, 0
// LLVM:   ret i1 %[[CMP]]

