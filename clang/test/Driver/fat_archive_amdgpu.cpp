// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// UNSUPPORTED: -aix

// See the steps to create a fat archive are given at the end of the file.

// Given a FatArchive, clang-offload-bundler should be called to create a
// device specific archive, which should be passed to llvm-link.
// RUN: %clang -O2 -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 %s -L%S/Inputs/openmp_static_device_link -lFatArchive 2>&1 | FileCheck %s
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm-bc"{{.*}}"-target-cpu" "[[GPU:gfx[0-9]+]]"{{.*}}"-o" "[[HOSTBC:.*.bc]]" "-x" "c++"{{.*}}.cpp
// CHECK: clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}/Inputs/openmp_static_device_link/libFatArchive.a" "-targets=openmp-amdgcn-amd-amdhsa-[[GPU]]" "-output=[[DEVICESPECIFICARCHIVE:.*.a]]" "-allow-missing-bundles"
// CHECK: llvm-link{{.*}}"[[HOSTBC]]" "[[DEVICESPECIFICARCHIVE]]" "-o" "{{.*}}-[[GPU]]-linked-{{.*}}.bc"
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#define N 10

#pragma omp declare target
// Functions defined in Fat Archive.
extern "C" void func_present(float *, float *, unsigned);

#ifdef MISSING
// Function not defined in the fat archive.
extern "C" void func_missing(float *, float *, unsigned);
#endif

#pragma omp end declare target

int main() {
  float in[N], out[N], sum = 0;
  unsigned i;

#pragma omp parallel for
  for (i = 0; i < N; ++i) {
    in[i] = i;
  }

  func_present(in, out, N); // Returns out[i] = a[i] * 0

#ifdef MISSING
  func_missing(in, out, N); // Should throw an error here
#endif

#pragma omp parallel for reduction(+ \
                                   : sum)
  for (i = 0; i < N; ++i)
    sum += out[i];

  if (!sum)
    return 0;
  return sum;
}

#endif

/***********************************************
   Steps to create Fat Archive (libFatArchive.a)
************************************************
***************** File: func_1.c ***************
void func_present(float* in, float* out, unsigned n){
  unsigned i;
  #pragma omp target teams distribute parallel for map(to: in[0:n]) map(from: out[0:n])
  for(i=0; i<n; ++i){
    out[i] = in[i] * 0;
  }
}
*************************************************
1. Compile source file(s) to generate object file(s)
    clang -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -c func_1.c -o func_1_gfx906.o
    clang -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -c func_1.c -o func_1_gfx908.o
    clang -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -c func_2.c -o func_2_gfx906.o
    clang -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -c func_2.c -o func_2_gfx908.o
    clang -O2 -fopenmp -fopenmp-targets=nvptx64 -c func_1.c -o func_1_nvptx.o
    clang -O2 -fopenmp -fopenmp-targets=nvptx64 -c func_2.c -o func_2_nvptx.o

2. Create a fat archive by combining all the object file(s)
    llvm-ar cr libFatArchive.a func_1_gfx906.o func_1_gfx908.o func_2_gfx906.o func_2_gfx908.o func_1_nvptx.o func_2_nvptx.o
************************************************/
