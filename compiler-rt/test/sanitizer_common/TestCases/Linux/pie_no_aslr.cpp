// RUN: %clangxx %s -pie -fPIE -o %t
// RUN: %if x86_64-target-arch %{ %run setarch x86_64 -R %t %}
// RUN: %if riscv64-target-arch %{ %run setarch rv64 -R %t %}
// REQUIRES: x86_64-target-arch || riscv64-target-arch

int main() { return 0; }
