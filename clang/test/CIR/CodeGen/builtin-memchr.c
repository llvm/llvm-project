// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void *test_memchr(const char arg[32]) {
  return __builtin_char_memchr(arg, 123, 32);
}

// CIR-LABEL: @test_memchr
// CIR: %[[PATTERN:.*]] = cir.const #cir.int<123> : !s32i
// CIR: %[[LEN:.*]] = cir.const #cir.int<32> : !u64i
// CIR: {{%.*}} = cir.libc.memchr({{%.*}}, %[[PATTERN]], %[[LEN]])

// LLVM-LABEL: @test_memchr
// LLVM: call ptr @memchr(ptr %{{.*}}, i32 123, i64 32)
// LLVM: ret ptr

// OGCG-LABEL: @test_memchr
// OGCG: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef 123, i64 noundef 32)
// OGCG: ret ptr
