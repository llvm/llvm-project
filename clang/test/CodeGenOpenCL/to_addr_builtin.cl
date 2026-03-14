// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -O0 -cl-std=clc++ -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -O0 -cl-std=cl2.0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -O0 -cl-std=cl3.0 -o - %s | FileCheck %s

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

  //CHECK: %[[ARG:.*]] = addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(1) %[[RET]], ptr %glob
  glob = to_global(glob);
  
  //CHECK: %[[ARG:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(1) %[[RET]], ptr %glob
  glob = to_global(loc);
 
  //CHECK: %[[ARG:.*]] = addrspacecast ptr %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(1) %[[RET]], ptr %glob
  glob = to_global(priv);
 
  //CHECK: %[[ARG:.*]] = load ptr addrspace(4), ptr %gen
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(1) %[[RET]], ptr %glob
  glob = to_global(gen);
  
  //CHECK: %[[ARG:.*]] = addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(3) @__to_local(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(3) %[[RET]], ptr %loc
  loc = to_local(glob);

  //CHECK: %[[ARG:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(3) @__to_local(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(3) %[[RET]], ptr %loc
  loc = to_local(loc);

  //CHECK: %[[ARG:.*]] = addrspacecast ptr %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(3) @__to_local(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(3) %[[RET]], ptr %loc
  loc = to_local(priv);

  //CHECK: %[[ARG:.*]] = load ptr addrspace(4), ptr %gen
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(3) @__to_local(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(3) %[[RET]], ptr %loc
  loc = to_local(gen);

  //CHECK: %[[ARG:.*]] = addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr @__to_private(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr %[[RET]], ptr %priv
  priv = to_private(glob);

  //CHECK: %[[ARG:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr @__to_private(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr %[[RET]], ptr %priv
  priv = to_private(loc);

  //CHECK: %[[ARG:.*]] = addrspacecast ptr %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr @__to_private(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr %[[RET]], ptr %priv
  priv = to_private(priv);

  //CHECK: %[[ARG:.*]] = load ptr addrspace(4), ptr %gen
  //CHECK: %[[RET:.*]] = call spir_func ptr @__to_private(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr %[[RET]], ptr %priv
  priv = to_private(gen);

  //CHECK: %[[ARG:.*]] = addrspacecast ptr %{{.*}} to ptr addrspace(4)
  //CHECK: %[[RET:.*]] = call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %[[ARG]])
  //CHECK: store ptr addrspace(1) %[[RET]], ptr %gA
  PA pA;
  GA gA = to_global(pA);

  //CHECK-NOT: addrspacecast
  //CHECK-NOT: bitcast
  //CHECK: call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %{{.*}})
  //CHECK-NOT: addrspacecast
  //CHECK-NOT: bitcast
  generic void *gen_v;
  global void *glob_v = to_global(gen_v);
}
