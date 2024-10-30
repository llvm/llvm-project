// This test checks if Window PE file compiled with -flto option contains a magic 
// string "LTCG" to indicate LTO compilation.

// REQUIRES: system-windows

// RUN: %clang --target=x86_64-pc-windows-msvc -flto -fuse-ld=lld %s -o %t.exe
// RUN: dumpbin /HEADERS %t.exe | FileCheck %s
// CHECK: {{.*}}LTCG{{.*}}

int main(void) {

	return 0;
}
