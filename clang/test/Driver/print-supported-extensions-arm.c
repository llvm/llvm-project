// REQUIRES: arm-registered-target
// RUN: %clang --target=arm-linux-gnu --print-supported-extensions | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s

// CHECK: All available -march extensions for ARM
// CHECK-EMPTY:
// CHECK-NEXT:     Name                Description
// CHECK-NEXT:     crc                 Enable support for CRC instructions
// CHECK-NEXT:     crypto              Enable support for Cryptography extensions
// CHECK-NEXT:     sha2                Enable SHA1 and SHA256 support
// CHECK-NEXT:     aes                 Enable AES support
// CHECK-NEXT:     dotprod             Enable support for dot product instructions
// CHECK-NEXT:     dsp                 Supports DSP instructions in ARM and/or Thumb2
// CHECK-NEXT:     mve                 Support M-Class Vector Extension with integer ops
// CHECK-NEXT:     mve.fp              Support M-Class Vector Extension with integer and floating ops
// CHECK-NEXT:     fp16                Enable half-precision floating point
// CHECK-NEXT:     ras                 Enable Reliability, Availability and Serviceability extensions
// CHECK-NEXT:     fp16fml             Enable full half-precision floating point fml instructions
// CHECK-NEXT:     bf16                Enable support for BFloat16 instructions
// CHECK-NEXT:     sb                  Enable v8.5a Speculation Barrier
// CHECK-NEXT:     i8mm                Enable Matrix Multiply Int8 Extension
// CHECK-NEXT:     lob                 Enable Low Overhead Branch extensions
// CHECK-NEXT:     cdecp0              Coprocessor 0 ISA is CDEv1
// CHECK-NEXT:     cdecp1              Coprocessor 1 ISA is CDEv1
// CHECK-NEXT:     cdecp2              Coprocessor 2 ISA is CDEv1
// CHECK-NEXT:     cdecp3              Coprocessor 3 ISA is CDEv1
// CHECK-NEXT:     cdecp4              Coprocessor 4 ISA is CDEv1
// CHECK-NEXT:     cdecp5              Coprocessor 5 ISA is CDEv1
// CHECK-NEXT:     cdecp6              Coprocessor 6 ISA is CDEv1
// CHECK-NEXT:     cdecp7              Coprocessor 7 ISA is CDEv1
// CHECK-NEXT:     pacbti              Enable Pointer Authentication and Branch Target Identification
