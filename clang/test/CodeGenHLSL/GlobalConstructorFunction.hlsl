// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

int i;

__attribute__((constructor)) void call_me_first(void) {
  i = 12;
}

__attribute__((constructor)) void then_call_me(void) {
  i = 12;
}

__attribute__((destructor)) void call_me_last(void) {
  i = 0;
}

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {}

// Make sure global variable for ctors/dtors removed.
// CHECK-NOT:@llvm.global_ctors
// CHECK-NOT:@llvm.global_dtors

//CHECK: define void @main()
//CHECK-NEXT: entry:
//CHECK-NEXT:   call void @"?call_me_first@@YAXXZ"()
//CHECK-NEXT:   call void @"?then_call_me@@YAXXZ"()
//CHECK-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
//CHECK-NEXT:   call void @"?main@@YAXI@Z"(i32 %0)
//CHECK-NEXT:   call void @"?call_me_last@@YAXXZ"(
//CHECK-NEXT:   ret void
