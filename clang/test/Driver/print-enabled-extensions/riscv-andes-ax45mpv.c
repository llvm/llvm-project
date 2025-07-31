// REQUIRES: riscv-registered-target
// RUN: %clang --target=riscv64 -mcpu=andes-ax45mpv --print-enabled-extensions | FileCheck %s

// CHECK: Extensions enabled for the given RISC-V target
// CHECK-EMPTY:
// CHECK-NEXT:     Name                 Version   Description
// CHECK-NEXT:     i                    2.1       'I' (Base Integer Instruction Set)
// CHECK-NEXT:     m                    2.0       'M' (Integer Multiplication and Division)
// CHECK-NEXT:     a                    2.1       'A' (Atomic Instructions)
// CHECK-NEXT:     f                    2.2       'F' (Single-Precision Floating-Point)
// CHECK-NEXT:     d                    2.2       'D' (Double-Precision Floating-Point)
// CHECK-NEXT:     c                    2.0       'C' (Compressed Instructions)
// CHECK-NEXT:     v                    1.0       'V' (Vector Extension for Application Processors)
// CHECK-NEXT:     zicsr                2.0       'Zicsr' (CSRs)
// CHECK-NEXT:     zifencei             2.0       'Zifencei' (fence.i)
// CHECK-NEXT:     zmmul                1.0       'Zmmul' (Integer Multiplication)
// CHECK-NEXT:     zaamo                1.0       'Zaamo' (Atomic Memory Operations)
// CHECK-NEXT:     zalrsc               1.0       'Zalrsc' (Load-Reserved/Store-Conditional)
// CHECK-NEXT:     zca                  1.0       'Zca' (part of the C extension, excluding compressed floating point loads/stores)
// CHECK-NEXT:     zcd                  1.0       'Zcd' (Compressed Double-Precision Floating-Point Instructions)
// CHECK-NEXT:     zve32f               1.0       'Zve32f' (Vector Extensions for Embedded Processors with maximal 32 EEW and F extension)
// CHECK-NEXT:     zve32x               1.0       'Zve32x' (Vector Extensions for Embedded Processors with maximal 32 EEW)
// CHECK-NEXT:     zve64d               1.0       'Zve64d' (Vector Extensions for Embedded Processors with maximal 64 EEW, F and D extension)
// CHECK-NEXT:     zve64f               1.0       'Zve64f' (Vector Extensions for Embedded Processors with maximal 64 EEW and F extension)
// CHECK-NEXT:     zve64x               1.0       'Zve64x' (Vector Extensions for Embedded Processors with maximal 64 EEW)
// CHECK-NEXT:     zvl128b              1.0       'Zvl128b' (Minimum Vector Length 128)
// CHECK-NEXT:     zvl32b               1.0       'Zvl32b' (Minimum Vector Length 32)
// CHECK-NEXT:     zvl64b               1.0       'Zvl64b' (Minimum Vector Length 64)
// CHECK-NEXT:     xandesperf           5.0       'XAndesPerf' (Andes Performance Extension)
// CHECK-EMPTY:
// CHECK-NEXT: Experimental extensions
// CHECK-EMPTY:
// CHECK-NEXT: ISA String: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xandesperf5p0
