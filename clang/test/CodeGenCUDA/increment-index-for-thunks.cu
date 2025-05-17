// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx942 \
// RUN:   -emit-llvm -xhip %s -o - | FileCheck %s --check-prefix=GCN
// RUN: %clang_cc1 -fcuda-is-device -triple spirv64-amd-amdhsa \
// RUN:   -emit-llvm -xhip %s -o - | FileCheck %s --check-prefix=SPIRV

// GCN: @_ZTV1C = linkonce_odr unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)], [4 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1B2f2Ev to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1C2f1Ev to ptr addrspace(1))], [4 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 -8 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZThn8_N1C2f1Ev to ptr addrspace(1))] }, comdat, align 8
// GCN: @_ZTV1B = linkonce_odr unnamed_addr addrspace(1) constant { [3 x ptr addrspace(1)] } { [3 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1B2f2Ev to ptr addrspace(1))] }, comdat, align 8
// GCN: @_ZTV1A = linkonce_odr unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @__cxa_pure_virtual to ptr addrspace(1))] }, comdat, align 8
// SPIRV: @_ZTV1C = linkonce_odr unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)], [4 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1B2f2Ev to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1C2f1Ev to ptr addrspace(1))], [4 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 -8 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZThn8_N1C2f1Ev to ptr addrspace(1))] }, comdat, align 8
// SPIRV: @_ZTV1B = linkonce_odr unnamed_addr addrspace(1) constant { [3 x ptr addrspace(1)] } { [3 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1B2f2Ev to ptr addrspace(1))] }, comdat, align 8
// SPIRV: @_ZTV1A = linkonce_odr unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr addrspace(4) @__cxa_pure_virtual to ptr addrspace(1))] }, comdat, align 8

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
