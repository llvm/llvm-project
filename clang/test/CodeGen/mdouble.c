// RUN: %clang_cc1 %s -emit-llvm -o - -triple=avr-unknown-unknown -mdouble=64 | \
// RUN:   FileCheck --check-prefix=AVR-FP64 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=avr-unknown-unknown -mdouble=32 | \
// RUN:   FileCheck --check-prefix=AVR-FP32 %s

double x = 0;
int size = sizeof(x);

// FIXME: the double should have an alignment of 1 on AVR, not 4 or 8.
// AVR-FP64: @x ={{.*}} global double {{.*}}, align 8
// AVR-FP64: @size ={{.*}} global i16 8
// AVR-FP32: @x ={{.*}} global float {{.*}}, align 4
// AVR-FP32: @size ={{.*}} global i16 4
