// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-X86-64
// RUN: %clang_cc1 -triple arm64-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM
// RUN: %clang_cc1 -triple armv7-linux-gnu -target-abi apcs-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM

// CHECK-NOT: @sp = common global

#if defined(__x86_64__)
register unsigned long current_stack_pointer asm("rsp");
#else
register unsigned long current_stack_pointer asm("sp");
#endif

struct p4_Thread {
  struct {
    int len;
  } word;
};
// Testing pointer types as well
#if defined(__x86_64__)
register struct p4_Thread *p4TH asm("rsp");
#else
register struct p4_Thread *p4TH asm("sp");
#endif

// CHECK: define{{.*}} i[[bits:[0-9]+]] @get_stack_pointer_addr()
// CHECK: [[ret:%[0-9]+]] = call i[[bits]] @llvm.read_register.i[[bits]](metadata !0)
// CHECK: ret i[[bits]] [[ret]]
unsigned long get_stack_pointer_addr() {
  return current_stack_pointer;
}
// CHECK: declare{{.*}} i[[bits]] @llvm.read_register.i[[bits]](metadata)

// CHECK: define{{.*}} void @set_stack_pointer_addr(i[[bits]] %addr) #0 {
// CHECK: [[sto:%[0-9]+]] = load i[[bits]], i[[bits]]* %
// CHECK: call void @llvm.write_register.i[[bits]](metadata !0, i[[bits]] [[sto]])
// CHECK: ret void
void set_stack_pointer_addr(unsigned long addr) {
  current_stack_pointer = addr;
}
// CHECK: declare{{.*}} void @llvm.write_register.i[[bits]](metadata, i[[bits]])

// CHECK: define {{.*}}@fn1
int fn1() {
  return (*p4TH).word.len;
}
// CHECK: %[[regr:[0-9]+]] = call i[[bits]] @llvm.read_register.i[[bits]](metadata !0)
// CHECK: inttoptr i[[bits]] %[[regr]] to %struct.p4_Thread*

// CHECK: define {{.*}}@fn2
void fn2(struct p4_Thread *val) {
  p4TH = val;
}
// CHECK: %[[regw:[0-9]+]] = ptrtoint %struct.p4_Thread* %{{.*}} to i[[bits]]
// CHECK: call void @llvm.write_register.i[[bits]](metadata !0, i[[bits]] %[[regw]])

// CHECK-X86-64: !llvm.named.register.rsp = !{!0}
// CHECK-X86-64: !0 = !{!"rsp"}
// CHECK-ARM: !llvm.named.register.sp = !{!0}
// CHECK-ARM: !0 = !{!"sp"}
