// RUN: %clang_cc1 -emit-llvm < %s

volatile unsigned char* const __attribute__((address_space(1))) serial_ctrl = 0x02;

