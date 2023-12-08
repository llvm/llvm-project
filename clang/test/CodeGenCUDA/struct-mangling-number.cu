// RUN: %clang_cc1 -emit-llvm -o - -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -triple amdgcn-amd-amdhsa \
// RUN:   -target-cpu gfx1030 -fcuda-is-device -x hip %s \
// RUN:   | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -aux-triple amdgcn-amd-amdhsa \
// RUN:   -aux-target-cpu gfx1030 -x hip %s \
// RUN:   | FileCheck -check-prefix=HOST %s

// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -aux-triple amdgcn-amd-amdhsa \
// RUN:   -aux-target-cpu gfx1030 -x hip %s \
// RUN:   | FileCheck -check-prefix=HOST-NEG %s

// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -x c++ %s \
// RUN:   | FileCheck -check-prefix=CPP %s

#if __HIP__
#include "Inputs/cuda.h"
#endif

// Check local struct 'Op' uses Itanium mangling number instead of MSVC mangling
// number in device side name mangling. It is the same in device and host
// compilation.

// DEV: define amdgpu_kernel void @_Z6kernelIZN4TestIiE3runEvE2OpEvv(

// HOST-DAG:     @{{.*}} = {{.*}}c"_Z6kernelIZN4TestIiE3runEvE2OpEvv\00"

// HOST-NEG-NOT: @{{.*}} = {{.*}}c"_Z6kernelIZN4TestIiE3runEvE2Op_1Evv\00"
#if __HIP__
template<typename T>
__attribute__((global)) void kernel()
{
}
#endif

// Check local struct 'Op' uses MSVC mangling number in host function name mangling.
// It is the same when compiled as HIP or C++ program.

// HOST-DAG: call void @"??$fun@UOp@?2??run@?$Test@H@@QEAAXXZ@@@YAXXZ"()
// CPP:      call void @"??$fun@UOp@?2??run@?$Test@H@@QEAAXXZ@@@YAXXZ"()
template<typename T>
void fun()
{
}

template <typename T>
class Test {
public:
  void run()
  {
    struct Op
    {
    };
#if __HIP__
    kernel<Op><<<1, 1>>>();
#endif
    fun<Op>();
  }
};

int main() {
  Test<int> A;
  A.run();
}
