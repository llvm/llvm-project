// This test checks if Windows PE file contains a "PGU" string to indicate that
// it was compiled using profiling data.

// REQUIRES: system-windows

// RUN: %clang --target=x86_64-pc-windows-msvc -fprofile-instr-generate="%profdata" -fuse-ld=lld %s -o %t.exe
// RUN: %t.exe
// RUN: llvm-profdata merge -output=%code.profdata %profdata
// RUN: %clang --target=x86_64-pc-windows-msvc -fprofile-use=%code.profdata -fuse-ld=lld %s -o %t.exe
// RUN: dumpbin /HEADERS %t.exe | FileCheck %s

// CHECK: {{.*}}PGU{{.*}}

int main(void) {

	return 0;
}
	
