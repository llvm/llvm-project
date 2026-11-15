// REQUIRES: riscv-registered-target
// RUN: %clang --target=riscv64 -mcpu=gaisler-gr765 --print-enabled-extensions | FileCheck %s
// CHECK: Extensions enabled for the given RISC-V target
// CHECK-EMPTY:
// CHECK-NEXT:     Name                 Version   Description
// CHECK-NEXT:     i                    2.1       'I' (Base Integer Instruction Set)
// CHECK-NEXT:     m                    2.0       'M' (Integer Multiplication and Division)
// CHECK-NEXT:     a                    2.1       'A' (Atomic Instructions)
// CHECK-NEXT:     f                    2.2       'F' (Single-Precision Floating-Point)
// CHECK-NEXT:     d                    2.2       'D' (Double-Precision Floating-Point)
// CHECK-NEXT:     c                    2.0       'C' (Compressed Instructions)
// CHECK-NEXT:     b                    1.0       'B' (the collection of the Zba, Zbb, Zbs extensions)
// CHECK-NEXT:     zic64b               1.0       'Zic64b' (Cache Block Size Is 64 Bytes)
// CHECK-NEXT:     zicbom               1.0       'Zicbom' (Cache-Block Management Instructions)
// CHECK-NEXT:     zicfiss              1.0       'Zicfiss' (Shadow stack)
// CHECK-NEXT:     zicntr               2.0       'Zicntr' (Base Counters and Timers)
// CHECK-NEXT:     zicond               1.0       'Zicond' (Integer Conditional Operations)
// CHECK-NEXT:     zicsr                2.0       'Zicsr' (CSRs)
// CHECK-NEXT:     zifencei             2.0       'Zifencei' (fence.i)
// CHECK-NEXT:     zihpm                2.0       'Zihpm' (Hardware Performance Counters)
// CHECK-NEXT:     zimop                1.0       'Zimop' (May-Be-Operations)
// CHECK-NEXT:     zmmul                1.0       'Zmmul' (Integer Multiplication)
// CHECK-NEXT:     zaamo                1.0       'Zaamo' (Atomic Memory Operations)
// CHECK-NEXT:     zalrsc               1.0       'Zalrsc' (Load-Reserved/Store-Conditional)
// CHECK-NEXT:     zfa                  1.0       'Zfa' (Additional Floating-Point)
// CHECK-NEXT:     zfbfmin              1.0       'Zfbfmin' (Scalar BF16 Converts)
// CHECK-NEXT:     zfh                  1.0       'Zfh' (Half-Precision Floating-Point)
// CHECK-NEXT:     zfhmin               1.0       'Zfhmin' (Half-Precision Floating-Point Minimal)
// CHECK-NEXT:     zca                  1.0       'Zca' (part of the C extension, excluding compressed floating point loads/stores)
// CHECK-NEXT:     zcb                  1.0       'Zcb' (Compressed basic bit manipulation instructions)
// CHECK-NEXT:     zcd                  1.0       'Zcd' (Compressed Double-Precision Floating-Point Instructions)
// CHECK-NEXT:     zcmop                1.0       'Zcmop' (Compressed May-Be-Operations)
// CHECK-NEXT:     zba                  1.0       'Zba' (Address Generation Instructions)
// CHECK-NEXT:     zbb                  1.0       'Zbb' (Basic Bit-Manipulation)
// CHECK-NEXT:     zbc                  1.0       'Zbc' (Carry-Less Multiplication)
// CHECK-NEXT:     zbkb                 1.0       'Zbkb' (Bitmanip instructions for Cryptography)
// CHECK-NEXT:     zbkc                 1.0       'Zbkc' (Carry-less multiply instructions for Cryptography)
// CHECK-NEXT:     zbkx                 1.0       'Zbkx' (Crossbar permutation instructions)
// CHECK-NEXT:     zbs                  1.0       'Zbs' (Single-Bit Instructions)
// CHECK-NEXT:     zkn                  1.0       'Zkn' (NIST Algorithm Suite)
// CHECK-NEXT:     zknd                 1.0       'Zknd' (NIST Suite: AES Decryption)
// CHECK-NEXT:     zkne                 1.0       'Zkne' (NIST Suite: AES Encryption)
// CHECK-NEXT:     zknh                 1.0       'Zknh' (NIST Suite: Hash Function Instructions)
// CHECK-NEXT:     zkt                  1.0       'Zkt' (Data Independent Execution Latency)
// CHECK-EMPTY:
// CHECK-NEXT: Experimental extensions
// CHECK-EMPTY:
// CHECK-NEXT: ISA String: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_b1p0_zic64b1p0_zicbom1p0_zicfiss1p0_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihpm2p0_zimop1p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfa1p0_zfbfmin1p0_zfh1p0_zfhmin1p0_zca1p0_zcb1p0_zcd1p0_zcmop1p0_zba1p0_zbb1p0_zbc1p0_zbkb1p0_zbkc1p0_zbkx1p0_zbs1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkt1p0
