// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -fopenmp -triple x86_64-pc-linux-gnu -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify -Wopenmp-target-exception -analyze
#pragma omp declare target
void foo(void) {
	throw 404;
}
#pragma omp end declare target
// expected-no-diagnostics
