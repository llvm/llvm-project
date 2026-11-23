// RUN: not %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml --target=arm-none-eabi -march=invalid
// RUN: not %clang -multi-lib-config=%S/Inputs/multilib/empty.yaml --target=aarch64-none-elf -march=invalid
