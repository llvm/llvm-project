// This test checks if COFF file compiled with
// -fprofile-generate has magic section ".pgi" to indicate so.

// RUN: %clang --target=x86_64-pc-windows-msvc -fprofile-generate %s -c -o %t_x86
// RUN: llvm-objdump -h %t_x86 | FileCheck --check-prefix=CHECK_PGI %s
// RUN: %clang --target=aarch64-pc-windows-msvc -fprofile-generate %s -c -o %t_aarch
// RUN: llvm-objdump -h %t_aarch | FileCheck --check-prefix=CHECK_PGI %s

// CHECK_PGI: {{.*}}.pgi{{.*}}

int main(void) {

	return 0;
}
