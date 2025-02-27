// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 < %s -o - | FileCheck %s
// CHECK:%struct.S = type { i32, i32 }
// CHECK:define{{.*}} void @test_addrspace(ptr addrspace(1) noundef %p1, ptr addrspace(2) noundef %p2) [[NUW:#[0-9]+]]
// CHECK:  [[p1addr:%.*]] = alloca ptr addrspace(1)
// CHECK:  [[p2addr:%.*]] = alloca ptr addrspace(2)
// CHECK:  store ptr addrspace(1) %p1, ptr [[p1addr]]
// CHECK:  store ptr addrspace(2) %p2, ptr [[p2addr]]
// CHECK:  [[t0:%.*]] = load ptr addrspace(2), ptr [[p2addr]], align 8 
// CHECK:  [[t1:%.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(2) [[t0]], i32 0, i32 1
// CHECK:  [[t2:%.*]] = load i32, ptr addrspace(2) [[t1]], align 4
// CHECK:  [[t3:%.*]] = load ptr addrspace(1), ptr [[p1addr]], align 8  
// CHECK:  [[t4:%.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(1) [[t3]], i32 0, i32 0 
// CHECK:  store i32 [[t2]], ptr addrspace(1) [[t4]], align 4
// CHECK:  [[t5:%.*]] = load ptr addrspace(2), ptr [[p2addr]], align 8  
// CHECK:  [[t6:%.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(2) [[t5]], i32 0, i32 0 
// CHECK:  [[t7:%.*]] = load i32, ptr addrspace(2) [[t6]], align 4            
// CHECK:  [[t8:%.*]] = load ptr addrspace(1), ptr [[p1addr]], align 8  
// CHECK:  [[t9:%.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(1) [[t8]], i32 0, i32 1 
// CHECK:  store i32 [[t7]], ptr addrspace(1) [[t9]], align 4
// CHECK:  ret void
// CHECK:}

// Check that we don't lose the address space when accessing a member
// of a structure.

#define __addr1    __attribute__((address_space(1)))
#define __addr2    __attribute__((address_space(2)))

typedef struct S {
  int a;
  int b;
} S;

void test_addrspace(__addr1 S* p1, __addr2 S*p2) {
  // swap
  p1->a = p2->b;
  p1->b = p2->a;
}

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
