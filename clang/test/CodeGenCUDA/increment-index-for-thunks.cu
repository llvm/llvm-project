// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx942 \
// RUN:   -emit-llvm -xhip %s -o - | FileCheck %s

// CHECK: @_ZTV1C = linkonce_odr unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)], [4 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1B2f2Ev to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1C2f1Ev to ptr addrspace(1))], [4 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 -8 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1C2f1Ev to ptr addrspace(1))] }, comdat, align 8
// CHECK: @_ZTV1B = linkonce_odr unnamed_addr addrspace(1) constant { [3 x ptr addrspace(1)] } { [3 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1B2f2Ev to ptr addrspace(1))] }, comdat, align 8
// CHECK: @_ZTV1A = linkonce_odr unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @__cxa_pure_virtual to ptr addrspace(1))] }, comdat, align 8
// CHECK: @__hip_cuid_ = addrspace(1) global i8 0
// CHECK: @llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_ to ptr)], section "llvm.metadata"
// CHECK: @__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

struct A {
  __attribute__((device)) A() { }
  virtual void neither_device_nor_host_f() = 0 ;
  __attribute__((device)) virtual void f1() = 0;
 
};
 
struct B {
  __attribute__((device)) B() { }
  __attribute__((device)) virtual void f2() { };
};
 
struct C : public B, public A {
  __attribute__((device)) C() : B(), A() { }
 
   virtual void neither_device_nor_host_f() override { }
  __attribute__((device)) virtual void f1() override { }
 
};
 
__attribute__((device)) void test() {
  C obj;
}
