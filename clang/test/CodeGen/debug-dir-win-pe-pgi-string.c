// This test checks if Windows PE file compiled with
// -fprofile-generate has magic string "PGI" to indicate so.


// REQUIRES: system-windows

// RUN: %clang --target=x86_64-pc-windows-msvc -fprofile-generate -fuse-ld=lld %s -o %t.exe
// RUN: dumpbin /HEADERS %t.exe | FileCheck --check-prefix=CHECK2 %s
// CHECK2: {{.*}}PGI{{.*}}

int main(void) {

	return 0;
}
