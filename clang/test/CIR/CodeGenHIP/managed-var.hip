// RUN: echo -n "GPU binary would be here." > %t

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fclangir \
// RUN:            -x hip -emit-cir %s -o %t.dev.cir
// RUN: FileCheck --check-prefix=CIR-DEV --input-file=%t.dev.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fclangir \
// RUN:            -x hip -emit-llvm %s -o %t.dev.cir.ll
// RUN: FileCheck --check-prefix=DEV --input-file=%t.dev.cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:            -x hip -emit-llvm %s -o %t.dev.ll
// RUN: FileCheck --check-prefix=DEV --input-file=%t.dev.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t \
// RUN:            -emit-cir %s -o %t.host.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.host.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t \
// RUN:            -emit-llvm %s -o %t.host.cir.ll
// RUN: FileCheck --check-prefix=HOST --input-file=%t.host.cir.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t \
// RUN:            -emit-llvm %s -o %t.host.ll
// RUN: FileCheck --check-prefix=HOST --input-file=%t.host.ll %s

#include "../CodeGenCUDA/Inputs/cuda.h"

__managed__ int x = 1;
__managed__ float arr[8];
extern __managed__ int ex;
static __managed__ int sx = 2;

template <typename T> __managed__ T tv = 3;

// CIR-DEV-DAG: cir.global external target_address_space(1) @x.managed = #cir.int<1> : !s32i align(4) {cu.externally_initialized = #cir.cu.externally_initialized, cu.var_registration = #cir.cu.var_registration<x, Variable, managed>}
// CIR-DEV-DAG: cir.global external target_address_space(1) @x = #cir.ptr<null> : !cir.ptr<!s32i, target_address_space(1)> {cu.externally_initialized = #cir.cu.externally_initialized}
// CIR-DEV-DAG: cir.global linkonce_odr comdat("_Z2tvIiE") target_address_space(1) @_Z2tvIiE.managed = #cir.int<3>

// DEV-DAG: @x.managed = addrspace(1) externally_initialized global i32 1, align 4
// DEV-DAG: @x = addrspace(1) externally_initialized global ptr addrspace(1) null
// DEV-DAG: @arr.managed = addrspace(1) externally_initialized global [8 x float] zeroinitializer, align 4
// DEV-DAG: @arr = addrspace(1) externally_initialized global ptr addrspace(1) null
// DEV-DAG: @ex.managed = external addrspace(1) global i32, align 4
// DEV-DAG: @ex = external addrspace(1) externally_initialized global ptr addrspace(1){{$}}
// DEV-DAG: @_ZL2sx.managed = addrspace(1) externally_initialized global i32 2, align 4
// DEV-DAG: @_ZL2sx = addrspace(1) externally_initialized global ptr addrspace(1) null
// DEV-DAG: @_Z2tvIiE.managed = linkonce_odr addrspace(1) externally_initialized global i32 3, comdat($_Z2tvIiE), align 4
// DEV-DAG: @_Z2tvIiE = linkonce_odr addrspace(1) externally_initialized global ptr addrspace(1) null{{$}}
// DEV-DAG: @llvm.compiler.used = {{.*}}@x.managed{{.*}}@x to{{.*}}@arr.managed{{.*}}@arr to{{.*}}@_ZL2sx.managed{{.*}}@_ZL2sx to{{.*}}@_Z2tvIiE.managed{{.*}}@_Z2tvIiE to

// HOST-DAG: @x.managed = internal global i32 1, align 4
// HOST-DAG: @x = internal externally_initialized global ptr null
// HOST-DAG: @arr.managed = internal global [8 x float] zeroinitializer, align 16
// HOST-DAG: @arr = internal externally_initialized global ptr null
// HOST-DAG: @ex.managed = external global i32, align 4
// HOST-DAG: @ex = external externally_initialized global ptr{{$}}
// HOST-DAG: @_ZL2sx.managed = internal global i32 2, align 4
// HOST-DAG: @_ZL2sx = internal externally_initialized global ptr null
// HOST-DAG: @_Z2tvIiE.managed = internal global i32 3, comdat($_Z2tvIiE), align 4
// HOST-DAG: @_Z2tvIiE = internal externally_initialized global ptr null

// CIR-DEV-LABEL: cir.func{{.*}} @_Z4kernPi
// CIR-DEV: %[[XADDR:.*]] = cir.get_global @x : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>, target_address_space(1)>
// CIR-DEV: %{{.*}} = cir.load align(4) %[[XADDR]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>, target_address_space(1)>, !cir.ptr<!s32i, target_address_space(1)>

// DEV-LABEL: define {{.*}}@_Z4kernPi
// DEV: %[[LD_X:.*]] = load ptr addrspace(1), ptr addrspace(1) @x, align 4
// DEV: store i32 2, ptr %{{.*}}, align 4
// DEV: load ptr addrspace(1), ptr addrspace(1) @arr, align 4
// DEV: load ptr addrspace(1), ptr addrspace(1) @ex, align 4
// DEV: load ptr addrspace(1), ptr addrspace(1) @_ZL2sx, align 4
// DEV: load ptr addrspace(1), ptr addrspace(1) @_Z2tvIiE, align 4
__attribute__((global)) void kern(int *out) {
  x = 2;
  arr[1] = 3.0f;
  *out = ex + sx + tv<int>;
}

// HOST-DAG: load ptr, ptr @x, align 4
// HOST-DAG: load ptr, ptr @ex, align 4
// HOST-DAG: load ptr, ptr @_ZL2sx, align 4
// HOST-DAG: load ptr, ptr @_Z2tvIiE, align 4
int host_use() { return x + ex + sx + tv<int>; }

// External managed variables are registered by their defining TU.
// CIR-HOST-LABEL: cir.func internal private @__hip_register_globals
// CIR-HOST-COUNT-4: cir.call @__hipRegisterManagedVar
// CIR-HOST-NEXT: cir.return

// HOST-DAG: call void @__hipRegisterManagedVar(ptr %{{.*}}, ptr @x, ptr @x.managed, ptr @{{.*}}, i64 4, i32 4)
// HOST-DAG: call void @__hipRegisterManagedVar(ptr %{{.*}}, ptr @arr, ptr @arr.managed, ptr @{{.*}}, i64 32, i32 16)
// HOST-DAG: call void @__hipRegisterManagedVar(ptr %{{.*}}, ptr @_ZL2sx, ptr @_ZL2sx.managed, ptr @{{.*}}, i64 4, i32 4)
// HOST-DAG: call void @__hipRegisterManagedVar(ptr %{{.*}}, ptr @_Z2tvIiE, ptr @_Z2tvIiE.managed, ptr @{{.*}}, i64 4, i32 4)
