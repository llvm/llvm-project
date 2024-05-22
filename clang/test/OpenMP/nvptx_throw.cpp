// REQUIRES: nvptx-registered-target, staticanalyzer

/**
 * The first four lines test that a warning is produced when enabling 
 * -Wopenmp-target-exception no matter what combination of -fexceptions and 
 * -fcxx-exceptions are set, as we want OpenMP to always allow exceptions in the
 * target region but emit a warning instead.
*/

// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device %s -emit-llvm -S -verify=with -Wopenmp-target-exception -analyze

/**
 * The following four lines test that no warning is emitted when providing 
 * -Wno-openmp-target-exception no matter the combination of -fexceptions and 
 * -fcxx-exceptions.
*/

// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fcxx-exceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fexceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device %s -emit-llvm -S -verify=without -Wno-openmp-target-exception -analyze

/**
 * Finally we should test that we only ignore exceptions in the OpenMP 
 * offloading tool-chain
*/

// RUN: %clang_cc1 -triple nvptx64 %s -emit-llvm -S -verify=noexceptions -o -

// noexceptions-error@37 {{cannot use 'throw' with exceptions disabled}}

#pragma omp declare target
void foo(void) {
	throw 404; // with-warning {{target 'nvptx64' does not support exception handling; 'throw' is assumed to be never reached}}
}
#pragma omp end declare target
// without-no-diagnostics
