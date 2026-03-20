// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=CHECK

[[vk::ext_extension("ext_on_global_var")]]
int global_val;

struct T
{
  [[vk::ext_extension("ext_on_field")]]
  int val;
};

struct [[vk::ext_extension("ext_on_struct")]] T2
{
  int val;
};

[[vk::ext_extension("ext_on_typedef")]]
typedef T MyTYpe;

[[vk::ext_extension("ext_on_cbuffer")]]
cbuffer cb { 
  int cb_val; 
};

[[vk::ext_extension("ext_on_function")]]
void foo([[vk::ext_extension("ext_on_param")]] int p) {}

[[vk::ext_extension("ext_on_entry_point")]]
[numthreads(1,1,1)]
void main() {
  T t;
  T2 t2;
  MyTYpe my_t;
  [[vk::ext_extension("ext_on_local_var")]]
  int local = global_val+t.val+t2.val+my_t.val+cb_val;
  foo(local);
}

// CHECK: !spirv.ext = !{!2, !3, !4, !5, !6, !7, !8, !9, !10}
// CHECK-DAG: !2 = !{!"ext_on_local_var"}
// CHECK-DAG: !3 = !{!"ext_on_field"}
// CHECK-DAG: !4 = !{!"ext_on_entry_point"}
// CHECK-DAG: !5 = !{!"ext_on_param"}
// CHECK-DAG: !6 = !{!"ext_on_function"}
// CHECK-DAG: !7 = !{!"ext_on_typedef"}
// CHECK-DAG: !8 = !{!"ext_on_struct"}
// CHECK-DAG: !9 = !{!"ext_on_cbuffer"}
// CHECK-DAG: !10 = !{!"ext_on_global_var"}
