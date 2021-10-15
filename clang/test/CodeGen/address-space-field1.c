// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 < %s -o - | FileCheck %s
// CHECK:%struct.S = type { i32, i32 }
// CHECK:define{{.*}} void @test_addrspace(%struct.S addrspace(1)* noundef %p1, %struct.S addrspace(2)* noundef %p2) [[NUW:#[0-9]+]]
// CHECK:  [[p1addr:%.*]] = alloca %struct.S addrspace(1)*
// CHECK:  [[p2addr:%.*]] = alloca %struct.S addrspace(2)*
// CHECK:  store %struct.S addrspace(1)* %p1, %struct.S addrspace(1)** [[p1addr]]
// CHECK:  store %struct.S addrspace(2)* %p2, %struct.S addrspace(2)** [[p2addr]]
// CHECK:  [[t0:%.*]] = load %struct.S addrspace(2)*, %struct.S addrspace(2)** [[p2addr]], align 8 
// CHECK:  [[t1:%.*]] = getelementptr inbounds %struct.S, %struct.S addrspace(2)* [[t0]], i32 0, i32 1
// CHECK:  [[t2:%.*]] = load i32, i32 addrspace(2)* [[t1]], align 4
// CHECK:  [[t3:%.*]] = load %struct.S addrspace(1)*, %struct.S addrspace(1)** [[p1addr]], align 8  
// CHECK:  [[t4:%.*]] = getelementptr inbounds %struct.S, %struct.S addrspace(1)* [[t3]], i32 0, i32 0 
// CHECK:  store i32 [[t2]], i32 addrspace(1)* [[t4]], align 4
// CHECK:  [[t5:%.*]] = load %struct.S addrspace(2)*, %struct.S addrspace(2)** [[p2addr]], align 8  
// CHECK:  [[t6:%.*]] = getelementptr inbounds %struct.S, %struct.S addrspace(2)* [[t5]], i32 0, i32 0 
// CHECK:  [[t7:%.*]] = load i32, i32 addrspace(2)* [[t6]], align 4            
// CHECK:  [[t8:%.*]] = load %struct.S addrspace(1)*, %struct.S addrspace(1)** [[p1addr]], align 8  
// CHECK:  [[t9:%.*]] = getelementptr inbounds %struct.S, %struct.S addrspace(1)* [[t8]], i32 0, i32 1 
// CHECK:  store i32 [[t7]], i32 addrspace(1)* [[t9]], align 4
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
