// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -O3 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x cuda -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x cuda -O3 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -O3 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x hip -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x hip -O3 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define ATTR __attribute__((device))
#else
#define ATTR
#endif

// Forced CUDA/HIP lambda numbering must be shared by all declarations of the same
// namespace. Otherwise, the second function pointer refers to the first lambda.
namespace reopened {
static ATTR int (*first)() = [] ATTR { return 144; };
}
namespace reopened {
static ATTR int (*second)() = [] ATTR { return 7; };
}

// CHECK-DAG: @_ZN8reopenedL5firstE = internal{{.*}} global ptr @_ZN8reopenedUlvE_8__invokeEv
// CHECK-DAG: @_ZN8reopenedL6secondE = internal{{.*}} global ptr @_ZN8reopenedUlvE0_8__invokeEv
// CHECK-DAG: define internal{{.*}} i32 @_ZN8reopenedUlvE_8__invokeEv()
// CHECK-DAG: define internal{{.*}} i32 @_ZN8reopenedUlvE0_8__invokeEv()
// OPT-LABEL: define{{.*}} i32 @named_namespace()
// OPT: ret i32 151
extern "C" ATTR int named_namespace() {
  return reopened::first() + reopened::second();
}

// Reopened anonymous namespaces also share the same numbering context.
namespace {
static ATTR int (*first)() = [] ATTR { return 20; };
}
namespace {
static ATTR int (*second)() = [] ATTR { return 3; };
}

// CHECK-DAG: @_ZN12_GLOBAL__N_15firstE = internal{{.*}} global ptr @_ZN12_GLOBAL__N_1UlvE_8__invokeEv
// CHECK-DAG: @_ZN12_GLOBAL__N_16secondE = internal{{.*}} global ptr @_ZN12_GLOBAL__N_1UlvE0_8__invokeEv
// CHECK-DAG: define internal{{.*}} i32 @_ZN12_GLOBAL__N_1UlvE_8__invokeEv()
// CHECK-DAG: define internal{{.*}} i32 @_ZN12_GLOBAL__N_1UlvE0_8__invokeEv()
// OPT-LABEL: define{{.*}} i32 @anonymous_namespace()
// OPT: ret i32 23
extern "C" ATTR int anonymous_namespace() { return first() + second(); }
