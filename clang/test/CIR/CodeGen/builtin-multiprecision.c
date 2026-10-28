// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

unsigned char test_addcb(unsigned char x, unsigned char y,
                         unsigned char carryin, unsigned char *carryout) {
  return __builtin_addcb(x, y, carryin, carryout);
}

// CIR-LABEL: cir.func{{.*}}@test_addcb
// CIR: cir.add.overflow %{{.+}}, %{{.+}} : !u8i -> !u8i
// CIR: cir.add.overflow %{{.+}}, %{{.+}} : !u8i -> !u8i
// CIR: cir.or %{{.+}}, %{{.+}} : !cir.bool
// CIR: cir.cast bool_to_int %{{.+}} : !cir.bool -> !u8i

// LLVM-LABEL: define {{.*}}@test_addcb
// LLVM: call { i8, i1 } @llvm.uadd.with.overflow.i8(i8 %{{.+}}, i8 %{{.+}})
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: call { i8, i1 } @llvm.uadd.with.overflow.i8(i8 %{{.+}}, i8 %{{.+}})
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: or i1 %{{.+}}, %{{.+}}
// LLVM: zext i1 %{{.+}} to i8
// LLVM: store i8 %{{.+}}, ptr %{{.+}}

unsigned int test_addc(unsigned int x, unsigned int y,
                       unsigned int carryin, unsigned int *carryout) {
  return __builtin_addc(x, y, carryin, carryout);
}

// CIR-LABEL: cir.func{{.*}}@test_addc
// CIR: cir.add.overflow %{{.+}}, %{{.+}} : !u32i -> !u32i
// CIR: cir.add.overflow %{{.+}}, %{{.+}} : !u32i -> !u32i
// CIR: cir.or %{{.+}}, %{{.+}} : !cir.bool
// CIR: cir.cast bool_to_int %{{.+}} : !cir.bool -> !u32i

// LLVM-LABEL: define {{.*}}@test_addc
// LLVM: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: or i1 %{{.+}}, %{{.+}}
// LLVM: zext i1 %{{.+}} to i32
// LLVM: store i32 %{{.+}}, ptr %{{.+}}

unsigned long long test_addcll(unsigned long long x, unsigned long long y,
                               unsigned long long carryin,
                               unsigned long long *carryout) {
  return __builtin_addcll(x, y, carryin, carryout);
}

// CIR-LABEL: cir.func{{.*}}@test_addcll
// CIR: cir.add.overflow %{{.+}}, %{{.+}} : !u64i -> !u64i
// CIR: cir.add.overflow %{{.+}}, %{{.+}} : !u64i -> !u64i
// CIR: cir.or %{{.+}}, %{{.+}} : !cir.bool
// CIR: cir.cast bool_to_int %{{.+}} : !cir.bool -> !u64i

// LLVM-LABEL: define {{.*}}@test_addcll
// LLVM: call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: or i1 %{{.+}}, %{{.+}}
// LLVM: zext i1 %{{.+}} to i64
// LLVM: store i64 %{{.+}}, ptr %{{.+}}

unsigned char test_subcb(unsigned char x, unsigned char y,
                         unsigned char carryin, unsigned char *carryout) {
  return __builtin_subcb(x, y, carryin, carryout);
}

// CIR-LABEL: cir.func{{.*}}@test_subcb
// CIR: cir.sub.overflow %{{.+}}, %{{.+}} : !u8i -> !u8i
// CIR: cir.sub.overflow %{{.+}}, %{{.+}} : !u8i -> !u8i
// CIR: cir.or %{{.+}}, %{{.+}} : !cir.bool
// CIR: cir.cast bool_to_int %{{.+}} : !cir.bool -> !u8i

// LLVM-LABEL: define {{.*}}@test_subcb
// LLVM: call { i8, i1 } @llvm.usub.with.overflow.i8(i8 %{{.+}}, i8 %{{.+}})
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: call { i8, i1 } @llvm.usub.with.overflow.i8(i8 %{{.+}}, i8 %{{.+}})
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i8, i1 } %{{.+}}, {{[01]}}
// LLVM: or i1 %{{.+}}, %{{.+}}
// LLVM: zext i1 %{{.+}} to i8
// LLVM: store i8 %{{.+}}, ptr %{{.+}}

unsigned int test_subc(unsigned int x, unsigned int y,
                       unsigned int carryin, unsigned int *carryout) {
  return __builtin_subc(x, y, carryin, carryout);
}

// CIR-LABEL: cir.func{{.*}}@test_subc
// CIR: cir.sub.overflow %{{.+}}, %{{.+}} : !u32i -> !u32i
// CIR: cir.sub.overflow %{{.+}}, %{{.+}} : !u32i -> !u32i
// CIR: cir.or %{{.+}}, %{{.+}} : !cir.bool
// CIR: cir.cast bool_to_int %{{.+}} : !cir.bool -> !u32i

// LLVM-LABEL: define {{.*}}@test_subc
// LLVM: call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i32, i1 } %{{.+}}, {{[01]}}
// LLVM: or i1 %{{.+}}, %{{.+}}
// LLVM: zext i1 %{{.+}} to i32
// LLVM: store i32 %{{.+}}, ptr %{{.+}}

unsigned long long test_subcll(unsigned long long x, unsigned long long y,
                               unsigned long long carryin,
                               unsigned long long *carryout) {
  return __builtin_subcll(x, y, carryin, carryout);
}

// CIR-LABEL: cir.func{{.*}}@test_subcll
// CIR: cir.sub.overflow %{{.+}}, %{{.+}} : !u64i -> !u64i
// CIR: cir.sub.overflow %{{.+}}, %{{.+}} : !u64i -> !u64i
// CIR: cir.or %{{.+}}, %{{.+}} : !cir.bool
// CIR: cir.cast bool_to_int %{{.+}} : !cir.bool -> !u64i

// LLVM-LABEL: define {{.*}}@test_subcll
// LLVM: call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: extractvalue { i64, i1 } %{{.+}}, {{[01]}}
// LLVM: or i1 %{{.+}}, %{{.+}}
// LLVM: zext i1 %{{.+}} to i64
// LLVM: store i64 %{{.+}}, ptr %{{.+}}
