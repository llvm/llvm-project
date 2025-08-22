// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// CHECK: [[ANNOT:.+]] = private unnamed_addr addrspace(1) constant {{.*}}c"my_annotation\00"

struct HasField {
  // This caused an assertion on creating a bitcast here,
  // since the address space didn't match.
  [[clang::annotate("my_annotation")]]
  int *a;
};

[[clang::sycl_external]] void foo(int *b) {
  struct HasField f;
  // CHECK: %[[A:.+]] = getelementptr inbounds nuw %struct.HasField, ptr addrspace(4) %{{.+}}
  // CHECK: %[[CALL:.+]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %[[A]], ptr addrspace(1) [[ANNOT]]
  // CHECK: store ptr addrspace(4) %{{[0-9]+}}, ptr addrspace(4) %[[CALL]]
  f.a = b;
}
