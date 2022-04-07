// RUN: %clang_cc1 -no-opaque-pointers -triple spir-unknown-unknown -emit-llvm -O0 -cl-std=clc++ -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple spir-unknown-unknown -emit-llvm -O0 -cl-std=cl2.0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple spir-unknown-unknown -emit-llvm -O0 -cl-std=cl3.0 -o - %s | FileCheck %s

// CHECK: %[[A:.*]] = type { float, float, float }
typedef struct {
  float x,y,z;
} A;
typedef private A *PA;
typedef global A *GA;

void test(void) {
  global int *glob;
  local int *loc;
  private int *priv;
  generic int *gen;

  //CHECK: %[[ARG:.*]] = addrspacecast i32 addrspace(1)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(1)* %[[RET]] to i32 addrspace(1)*
  glob = to_global(glob);
  
  //CHECK: %[[ARG:.*]] = addrspacecast i32 addrspace(3)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(1)* %[[RET]] to i32 addrspace(1)*
  glob = to_global(loc);
 
  //CHECK: %[[ARG:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(1)* %[[RET]] to i32 addrspace(1)*
  glob = to_global(priv);
 
  //CHECK: %[[ARG:.*]] = bitcast i32 addrspace(4)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(1)* %[[RET]] to i32 addrspace(1)*
  glob = to_global(gen);
  
  //CHECK: %[[ARG:.*]] = addrspacecast i32 addrspace(1)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(3)* @__to_local(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(3)* %[[RET]] to i32 addrspace(3)*
  loc = to_local(glob);

  //CHECK: %[[ARG:.*]] = addrspacecast i32 addrspace(3)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(3)* @__to_local(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(3)* %[[RET]] to i32 addrspace(3)*
  loc = to_local(loc);

  //CHECK: %[[ARG:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(3)* @__to_local(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(3)* %[[RET]] to i32 addrspace(3)*
  loc = to_local(priv);

  //CHECK: %[[ARG:.*]] = bitcast i32 addrspace(4)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(3)* @__to_local(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(3)* %[[RET]] to i32 addrspace(3)*
  loc = to_local(gen);

  //CHECK: %[[ARG:.*]] = addrspacecast i32 addrspace(1)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8* @__to_private(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8* %[[RET]] to i32*
  priv = to_private(glob);

  //CHECK: %[[ARG:.*]] = addrspacecast i32 addrspace(3)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8* @__to_private(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8* %[[RET]] to i32*
  priv = to_private(loc);

  //CHECK: %[[ARG:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8* @__to_private(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8* %[[RET]] to i32*
  priv = to_private(priv);

  //CHECK: %[[ARG:.*]] = bitcast i32 addrspace(4)* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8* @__to_private(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8* %[[RET]] to i32*
  priv = to_private(gen);

  //CHECK: %[[ARG:.*]] = addrspacecast %[[A]]* %{{.*}} to i8 addrspace(4)*
  //CHECK: %[[RET:.*]] = call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %[[ARG]])
  //CHECK: %{{.*}} = bitcast i8 addrspace(1)* %[[RET]] to %[[A]] addrspace(1)*
  PA pA;
  GA gA = to_global(pA);

  //CHECK-NOT: addrspacecast
  //CHECK-NOT: bitcast
  //CHECK: call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %{{.*}})
  //CHECK-NOT: addrspacecast
  //CHECK-NOT: bitcast
  generic void *gen_v;
  global void *glob_v = to_global(gen_v);
}
