// RUN: %clang_cc1 -no-opaque-pointers -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// CHECK: [[ANNOT:.+]] = private unnamed_addr addrspace(1) constant {{.*}}c"my_annotation\00"

struct HasField {
  // This caused an assertion on creating a bitcast here,
  // since the address space didn't match.
  [[clang::annotate("my_annotation")]]
  int *a;
};

void foo(int *b) {
  struct HasField f;
  // CHECK: %[[A:.+]] = getelementptr inbounds %struct.HasField, %struct.HasField addrspace(4)* %{{.+}}
  // CHECK: %[[BITCAST:.+]] = bitcast i32 addrspace(4)* addrspace(4)* %[[A]] to i8 addrspace(4)*
  // CHECK: %[[CALL:.+]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 addrspace(4)* %[[BITCAST]], i8 addrspace(1)* getelementptr inbounds ([14 x i8], [14 x i8] addrspace(1)* [[ANNOT]]
  // CHECK: bitcast i8 addrspace(4)* %[[CALL]] to i32 addrspace(4)* addrspace(4)*
  f.a = b;
}
