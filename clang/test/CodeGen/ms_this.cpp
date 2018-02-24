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

// CHECK: define dso_local void @"\01?runc@t2@@
void t2::runc() {
  double num = 0;
  __asm {
      mov rax,[this]
      // CHECK: [[THIS_ADDR_T2:%.+]] = alloca %class.t2*
      // CHECK: [[THIS1_T2:%.+]] = load %class.t2*, %class.t2** [[THIS_ADDR_T2]],
      // CHECK: call void asm sideeffect inteldialect "mov rax,qword ptr $1{{.*}}%class.t2* [[THIS1_T2]]
      mov rbx,[rax]
      mov num, rbx
	   };
}

// CHECK: define dso_local void @"\01?runc@t1@@
void t1::runc() {
  double num = 0;
  __asm {
       mov rax,[this]
       // CHECK: [[THIS_ADDR_T1:%.+]] = alloca %class.t1*
       // CHECK: [[THIS1_T1:%.+]] = load %class.t1*, %class.t1** [[THIS_ADDR_T1]],
       // CHECK: call void asm sideeffect inteldialect "mov rax,qword ptr $1{{.*}}%class.t1* [[THIS1_T1]]
        mov rbx,[rax]
        mov num, rbx
	   };
}

struct s {
  int a;
  // CHECK: define linkonce_odr dso_local void @"\01?func@s@@
  void func() {
    __asm mov rax, [this]
    // CHECK: [[THIS_ADDR_S:%.+]] = alloca %struct.s*
    // CHECK: [[THIS1_S:%.+]] = load %struct.s*, %struct.s** [[THIS_ADDR_S]],
    // CHECK: call void asm sideeffect inteldialect "mov rax, qword ptr $0{{.*}}%struct.s* [[THIS1_S]]
  }
} f3;

int main() {
  f3.func();
  f3.a=1;
  return 0;
}
