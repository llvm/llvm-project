// RUN: %clang_cc1 -triple spirv64-intel %s -emit-llvm -o - | FileCheck %s

// Test that function pointer casts properly handle address space conversions
// on targets like spirv64-intel that use a non-default program address space.

void foo() {}

// CHECK-LABEL: define spir_func void @_Z21test_func_to_void_ptrv() addrspace(9)
void test_func_to_void_ptr() {
    void *ptr = (void*)foo;
    // CHECK: store ptr addrspace(4) addrspacecast (ptr addrspace(9) @_Z3foov to ptr addrspace(4))
}

// CHECK-LABEL: define spir_func void @_Z21test_void_ptr_to_funcv() addrspace(9)
void test_void_ptr_to_func() {
    void *ptr = (void*)foo;
    void (*fptr)() = (void (*)())ptr;
    // CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(9)
    fptr();
}

// CHECK-LABEL: define spir_func void @_Z25cxx_test_func_to_void_ptrv() addrspace(9)
void cxx_test_func_to_void_ptr() {
  void *ptr = reinterpret_cast<void*>(foo);
  // CHECK: store ptr addrspace(4) addrspacecast (ptr addrspace(9) @_Z3foov to ptr addrspace(4))
}

// CHECK-LABEL: define spir_func void @_Z25cxx_test_void_ptr_to_funcv() addrspace(9)
void cxx_test_void_ptr_to_func() {
  void *ptr = reinterpret_cast<void*>(foo);
  void (*fptr)() = reinterpret_cast<void (*)()>(ptr);
  // CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(9)
  fptr();
}
