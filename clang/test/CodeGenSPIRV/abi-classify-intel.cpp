// Checks how SPIR-V functions on the Intel vendor triple pass and return
// values. The vendor makes the Generic address space the default without a
// device language, so kernels coerce Generic pointers but don't copy
// non-trivial records byval.

// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64-intel -fclangir -emit-llvm %s -o - | FileCheck %s %}

struct Small { int a, b; };
struct NonTrivialCopy { int i; NonTrivialCopy(const NonTrivialCopy &); };
struct NoCopy {
  NoCopy();
  NoCopy(const NoCopy &);
  ~NoCopy();
  int a;
};

// Pointers keep their address space outside kernels.
// CHECK: define spir_func void @_Z7dev_ptrPi(ptr addrspace(4) noundef %{{[^,)]+}})
void dev_ptr(int *p) {}

// A record that cannot be copied is built in place through a Generic sret.
// CHECK-LABEL: define spir_func void @_Z11ret_no_copyv(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NoCopy) align 4 %[[RET:[^,)]+]])
// CHECK:   call spir_func addrspace(9) void @_ZN6NoCopyC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %[[RET]])
NoCopy ret_no_copy() { return NoCopy(); }

// CHECK-LABEL: define spir_func void @_Z11use_no_copyv()
// CHECK:   call spir_func addrspace(9) void @_Z11ret_no_copyv(ptr addrspace(4) dead_on_unwind writable sret(%struct.NoCopy) align 4 %{{[^,)]+}})
void use_no_copy() { NoCopy x = ret_no_copy(); }

// Generic kernel pointers are passed as CrossWorkgroup ones; the body still
// sees a Generic pointer.
// CHECK-LABEL: define spir_kernel void @_Z5k_ptrPi(ptr addrspace(1) noundef %{{[^,)]+}})
// CHECK:   call spir_func addrspace(9) void @_Z7dev_ptrPi(ptr addrspace(4) noundef %{{[^,)]+}})
__attribute__((device_kernel)) void k_ptr(int *p) { dev_ptr(p); }

// Pointers in any other address space are left alone, including
// std::nullptr_t, which is a private pointer.
// CHECK: define spir_kernel void @_Z13k_private_ptrPU3AS0i(ptr noundef %{{[^,)]+}})
__attribute__((device_kernel)) void k_private_ptr(__attribute__((address_space(0))) int *p) {}

// CHECK: define spir_kernel void @_Z9k_nullptrDn(ptr %{{[^,)]+}})
__attribute__((device_kernel)) void k_nullptr(decltype(nullptr) n) {}

// Kernels copy trivially copyable records byval, but pass a non-trivial one
// as the address of the caller's copy.
// CHECK: define spir_kernel void @_Z7k_small5Small(ptr noundef byval(%struct.Small) align 4 %{{[^,)]+}})
__attribute__((device_kernel)) void k_small(Small s) {}

// CHECK: define spir_kernel void @_Z18k_non_trivial_copy14NonTrivialCopy(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{[^,)]+}})
__attribute__((device_kernel)) void k_non_trivial_copy(NonTrivialCopy n) {}
