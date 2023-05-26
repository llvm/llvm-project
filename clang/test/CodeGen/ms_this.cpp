// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-win32 -fasm-blocks -emit-llvm %s -o - | FileCheck %s
class t1 {
public:
  double a;
  void runc();
};

class t2 {
public:
  double a;
  void runc();
};

// CHECK: define dso_local void @"?runc@t2@@
void t2::runc() {
  double num = 0;
  __asm {
      mov rax,[this]
      // CHECK: [[THIS_ADDR_T2:%.+]] = alloca ptr
      // CHECK: [[THIS1_T2:%.+]] = load ptr, ptr [[THIS_ADDR_T2]],
      // CHECK: call void asm sideeffect inteldialect "mov rax,$1\0A\09mov rbx,[rax]\0A\09mov $0, rbx", "=*m,m,~{rax},~{rbx},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(double) %num, ptr [[THIS1_T2]])
      mov rbx,[rax]
      mov num, rbx
	   };
}

// CHECK: define dso_local void @"?runc@t1@@
void t1::runc() {
  double num = 0;
  __asm {
       mov rax,[this]
       // CHECK: [[THIS_ADDR_T1:%.+]] = alloca ptr
       // CHECK: [[THIS1_T1:%.+]] = load ptr, ptr [[THIS_ADDR_T1]],
       // CHECK: call void asm sideeffect inteldialect "mov rax,$1{{.*}}ptr [[THIS1_T1]]
        mov rbx,[rax]
        mov num, rbx
	   };
}

struct s {
  int a;
  // CHECK: define linkonce_odr dso_local void @"?func@s@@
  void func() {
    __asm mov rax, [this]
    // CHECK: [[THIS_ADDR_S:%.+]] = alloca ptr
    // CHECK: [[THIS1_S:%.+]] = load ptr, ptr [[THIS_ADDR_S]],
    // CHECK: call void asm sideeffect inteldialect "mov rax, $0{{.*}}ptr [[THIS1_S]]
  }
} f3;

int main() {
  f3.func();
  f3.a=1;
  return 0;
}
