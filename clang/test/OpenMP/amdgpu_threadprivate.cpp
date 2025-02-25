// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -target-cpu gfx906 -fopenmp -nogpulib -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck %s

// Don't crash with assertions build.

// CHECK:       @MyGlobVar = external thread_local addrspace(1) global i32, align 4
// CHECK:       define weak_odr hidden noundef ptr @_ZTW9MyGlobVar() #0 comdat {
// CHECK-NEXT:    %1 = call align 4 ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) align 4 @MyGlobVar)
// CHECK-NEXT:    %2 = addrspacecast ptr addrspace(1) %1 to ptr
// CHECK-NEXT:    ret ptr %2
// CHECK-NEXT:  }
int MyGlobVar;
#pragma omp threadprivate(MyGlobVar)
int main() {
  MyGlobVar = 1;
}

