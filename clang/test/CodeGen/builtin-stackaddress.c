// RUN: %clang_cc1 -triple i686-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix=X86 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix=X86_64 %s

void* a() {
  // X86_64: [[INT_SP:%.*]] = call i64 @llvm.read_register.i64(metadata [[SPREG:![0-9]+]])
  // X86_64: inttoptr i64 [[INT_SP]]
  // X86_64: [[SPREG]] = !{!"rsp"}
  //
  // X86: [[INT_SP:%.*]] = call i32 @llvm.read_register.i32(metadata [[SPREG:![0-9]+]])
  // X86: inttoptr i32 [[INT_SP]]
  // X86: [[SPREG]] = !{!"esp"}
  return __builtin_stack_address();
}
