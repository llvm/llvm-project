// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -w -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -w -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -target-cpu gfx906 -o - | FileCheck %s
// expected-no-diagnostics


/*===-----------------------------------------------------------------------=== 

Inspired from SOLLVE tests:
 - 5.0/metadirective/test_metadirective_arch_is_nvidia.c


===------------------------------------------------------------------------===*/


#define N 1024

int metadirective1() {
   
   int v1[N], v2[N], v3[N];

   int target_device_num, host_device_num, default_device;
   int errors = 0;

   #pragma omp target map(to:v1,v2) map(from:v3, target_device_num) device(default_device)
   {
      #pragma omp metadirective \
                   when(device={arch("amdgcn")}: teams distribute parallel for) \
                   default(parallel for)

         for (int i = 0; i < N; i++) {
	    #pragma omp atomic write
            v3[i] = v1[i] * v2[i];
         }
   }

   return errors;
}

// CHECK-LABEL: define weak_odr amdgpu_kernel void {{.+}}metadirective1
// CHECK: entry:
// CHECK: %{{[0-9]}} = call i32 @__kmpc_target_init
// CHECK: user_code.entry:
// CHECK: call void @__omp_outlined__
// CHECK-NOT: call void @__kmpc_parallel_51
// CHECK: ret void


// CHECK-LABEL: define internal void @__omp_outlined__
// CHECK: entry:
// CHECK: call void @__kmpc_distribute_static_init
// CHECK: omp.loop.exit:  
// CHECK: call void @__kmpc_distribute_static_fini


// CHECK-LABEL: define internal void @__omp_outlined__.{{[0-9]+}}
// CHECK: entry:
// CHECK: call void @__kmpc_for_static_init_4
// CHECK: omp.inner.for.body:
// CHECK: store atomic {{.*}} monotonic
// CHECK: omp.loop.exit:                                    
// CHECK-NEXT: call void @__kmpc_distribute_static_fini
// CHECK-NEXT: ret void

