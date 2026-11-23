// RUN: %clang -target x86_64-pc-win32  -ffile-reproducible -emit-llvm -S -g %s -o - | FileCheck --check-prefix=WIN %s
// RUN: %clang -target x86_64-linux-gnu  -ffile-reproducible -emit-llvm -S -g %s -o - | FileCheck --check-prefix=LINUX %s
int main() { return 0; }

// WIN:   !DIFile(filename: "{{.*}}\\slash.c"
// LINUX: !DIFile(filename: "{{.*}}/slash.c"
