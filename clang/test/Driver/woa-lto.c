// REQUIRES: aarch64-registered-target
//
// RUN: echo "int main() {} " > %t.c
//
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto -fuse-ld=lld -c %t.c -o %t.o
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto -fuse-ld=lld -### %t.o 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto=thin -fuse-ld=lld -c %t.c -o %t.o
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto=thin -fuse-ld=lld -### %t.o 2>&1 | FileCheck %s
//
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto -fuse-ld=lld-link -c %t.c -o %t.o
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto -fuse-ld=lld-link -### %t.o 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto=thin -fuse-ld=lld-link -c %t.c -o %t.o
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto=thin -fuse-ld=lld-link -### %t.o 2>&1 | FileCheck %s
//
// CHECK: "{{.*}}lld-link{{(.exe)?}}" "-out:a.exe" "-defaultlib:libcmt" "-defaultlib:oldnames"
