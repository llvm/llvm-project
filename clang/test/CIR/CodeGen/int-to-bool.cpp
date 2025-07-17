// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

bool f1(unsigned char c) {
  return c;
}

// CIR: cir.func{{.*}} @_Z2f1h
// CIR:   cir.cast(int_to_bool, %{{.*}} : !u8i), !cir.bool

// Note: The full zext/store/load/trunc sequence is checked here to show what
// CIR is being lowered to. There's no need to check it for every function since
// the lowering is the same for all of them.

// LLVM: define{{.*}} i1 @_Z2f1h
// LLVM:   %[[CMP:.*]] = icmp ne i8 %4, 0
// LLVM:   %[[ZEXT:.*]] = zext i1 %[[CMP]] to i8
// LLVM:   store i8 %[[ZEXT]], ptr %{{.*}}
// LLVM:   %[[LOAD:.*]] = load i8, ptr %{{.*}}
// LLVM:   %[[TRUNC:.*]] = trunc i8 %[[LOAD]] to i1
// LLVM:   ret i1 %[[TRUNC]]

// OGCG: define{{.*}} i1 @_Z2f1h
// OGCG:   %[[CMP:.*]] = icmp ne i8 %{{.*}}, 0
// OGCG:   ret i1 %[[CMP]]

bool f2(short s) {
  return s;
}

// CIR: cir.func{{.*}} @_Z2f2s
// CIR:   cir.cast(int_to_bool, %{{.*}} : !s16i), !cir.bool

// LLVM: define{{.*}} i1 @_Z2f2s
// LLVM:   %[[CMP:.*]] = icmp ne i16 %4, 0
// LLVM:   %[[ZEXT:.*]] = zext i1 %[[CMP]] to i8

// OGCG: define{{.*}} i1 @_Z2f2s
// OGCG:   %[[CMP:.*]] = icmp ne i16 %{{.*}}, 0
// OGCG:   ret i1 %[[CMP]]

bool f3(unsigned u) {
  return u;
}

// CIR: cir.func{{.*}} @_Z2f3j
// CIR:   cir.cast(int_to_bool, %{{.*}} : !u32i), !cir.bool

// LLVM: define{{.*}} i1 @_Z2f3j
// LLVM:   %[[CMP:.*]] = icmp ne i32 %4, 0
// LLVM:   %[[ZEXT:.*]] = zext i1 %[[CMP]] to i8

// OGCG: define{{.*}} i1 @_Z2f3j
// OGCG:   %[[CMP:.*]] = icmp ne i32 %{{.*}}, 0
// OGCG:   ret i1 %[[CMP]]

bool f4(long l) {
  return l;
}

// CIR: cir.func{{.*}} @_Z2f4l
// CIR:   cir.cast(int_to_bool, %{{.*}} : !s64i), !cir.bool

// LLVM: define{{.*}} i1 @_Z2f4l
// LLVM:   %[[CMP:.*]] = icmp ne i64 %4, 0
// LLVM:   %[[ZEXT:.*]] = zext i1 %[[CMP]] to i8

// OGCG: define{{.*}} i1 @_Z2f4l
// OGCG:   %[[CMP:.*]] = icmp ne i64 %{{.*}}, 0
// OGCG:   ret i1 %[[CMP]]
