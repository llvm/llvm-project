// RUN: %clang_cc1 -emit-llvm -o - -triple i686-pc-darwin %s | FileCheck -check-prefix=X86 %s
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn %s | FileCheck -check-prefix=AMDGCN %s
struct A {
  int x[100];
};

int f(struct A a);

int g() {
  struct A a;
  // X86:    call i32 @f(ptr noundef nonnull byval(%struct.A) align 4 %a)
  // AMDGCN: call i32 @f(ptr addrspace(5) noundef byref{{.*}}%a)
  return f(a);
}

// X86:   declare i32 @f(ptr noundef byval(%struct.A) align 4)
// AMDGCN: declare i32 @f(ptr addrspace(5) noundef byref{{.*}})
