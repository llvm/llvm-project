// This test checks if COFF file compiled with
// -fprofile-generate has magic section ".pgi" to indicate so.

// REQUIRES: x86-registered-target

// RUN: %clang --target=x86_64-pc-windows -fprofile-generate %s -c -o %t_x86
// RUN: llvm-objdump -h %t_x86 | FileCheck --check-prefix=CHECK_PGI %s

// CHECK_PGI: {{.*}}.pgi{{.*}}

// This test checks if COFF file contains a magic ".pgu" section to indicate that
// it was compiled using profiling data.

// RUN: llvm-profdata merge -output=%code.profdata %S/Inputs/thinlto_expect1.proftext
// RUN: %clang --target=x86_64-pc-windows -fprofile-use=%code.profdata -c %s -o %t.obj
// RUN: llvm-objdump -h %t.obj | FileCheck --check-prefix=CHECK_PGU %s

// CHECK_PGU: {{.*}}.pgu{{.*}}

int main(void) {

	return 0;
}
