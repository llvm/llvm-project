// RUN: %clang_cc1 -triple spirv64 -fobjc-runtime=gnustep-2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,PRIV
// RUN: %clang_cc1 -triple spirv64-intel -fobjc-runtime=gnustep-2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GEN

// Objective-C object pointers are lowered in the target address space of the
// default address space, like other pointers. On spirv64-intel that is the
// generic address space, so converting between id and void * needs no cast.

@interface Foo
@end

// PRIV: @glob = global ptr null
// GEN:  @glob = addrspace(1) global ptr addrspace(4) null
id glob;

// PRIV: define spir_func ptr @to_void(ptr noundef %x)
// GEN:  define spir_func ptr addrspace(4) @to_void(ptr addrspace(4) noundef %x)
// CHECK:      [[V:%.*]] = load
// CHECK-NEXT: ret {{.*}} [[V]]
void *to_void(id x) { return (void *)x; }

// PRIV: define spir_func ptr @from_void(ptr noundef %p)
// GEN:  define spir_func ptr addrspace(4) @from_void(ptr addrspace(4) noundef %p)
// CHECK:      [[V:%.*]] = load
// CHECK-NEXT: ret {{.*}} [[V]]
id from_void(void *p) { return (id)p; }

// PRIV: define spir_func ptr @to_foo(ptr noundef %x)
// GEN:  define spir_func ptr addrspace(4) @to_foo(ptr addrspace(4) noundef %x)
Foo *to_foo(id x) { return x; }
