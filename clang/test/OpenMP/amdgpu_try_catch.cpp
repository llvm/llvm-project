// REQUIRES: amdgpu-registered-target

/**
 * The first four lines test that a warning is produced when enabling 
 * -Wopenmp-target-exception no matter what combination of -fexceptions and 
 * -fcxx-exceptions are set, as we want OpenMP to always allow exceptions in the
 * target region but emit a warning instead.
*/

// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze

/**
 * The following four lines test that no warning is emitted when providing 
 * -Wno-openmp-target-exception no matter the combination of -fexceptions and 
 * -fcxx-exceptions.
*/

// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fcxx-exceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fexceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze

/**
 * Finally we should test that we only ignore exceptions in the OpenMP 
 * offloading tool-chain
*/

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa %s -emit-llvm -S -verify=noexceptions -o -

// noexceptions-error@38 {{cannot use 'try' with exceptions disabled}}

#pragma omp declare target
int foo(void) {
	int error = -1;
	try { // with-warning {{target 'amdgcn-amd-amdhsa' does not support exception handling; 'catch' block is ignored}}
		error = 1;
	}
	catch (int e){ 
		error = e;
	}
	return error;
}
#pragma omp end declare target
// without-no-diagnostics
