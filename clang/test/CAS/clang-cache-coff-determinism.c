// REQUIRES: ondisk_cas
// REQUIRES: system-windows

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang-cache %clang -o %t/test.obj.1 %t/test.c
// RUN: %clang-cache %clang -o %t/test.obj.2 %t/test.c -Xclang -fcache-disable-replay
// RUN: diff %t/test.obj.1 %t/test.obj.2

//--- test.c
int main() { return 0; }
