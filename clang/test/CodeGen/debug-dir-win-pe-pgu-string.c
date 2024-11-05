// This test checks if Windows PE file contains a "PGU" string to indicate that
// it was compiled using profiling data.

// RUN: llvm-profdata merge -output=%code.profdata %S/Inputs/thinlto_expect1.proftext
// RUN: %clang --target=x86_64-pc-windows -fprofile-use=%code.profdata -c %s -o %t.obj
// RUN: llvm-objdump -h %t.obj | FileCheck --check-prefix=CHECK_PGU %s

// RUN: %clang --target=aarch64-windows -fprofile-use=%code.profdata -c %s -o %t.obj
// RUN: llvm-objdump -h %t.obj | FileCheck --check-prefix=CHECK_PGU %s


// CHECK_PGU: {{.*}}.pgu{{.*}}

int main(void) {

	return 0;
}
	
