// RUN: %clangxx %s -pie -fPIE -o %t && %run setarch x86_64 -R %t
// REQUIRES: x86_64-target-arch

int main() { return 0; }
