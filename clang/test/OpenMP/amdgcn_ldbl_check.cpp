// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple x86_64-mingw64 -emit-llvm-bc -target-cpu x86-64 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -o %t.bc -x c++ %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-mingw64 -fsyntax-only -target-cpu gfx900 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.bc -x c++ %s
// expected-no-diagnostics

void print(double);

constexpr double operator"" _X (long double a)
{
	return (double)a;
}

int main()
{
	auto a = 1._X;
  print(a);
#pragma omp target map(tofrom: a)
	{
#pragma omp teams num_teams(1) thread_limit(4)
		{
			a += 1._X;
		}
	}
  print(a);
	return 0;
}
