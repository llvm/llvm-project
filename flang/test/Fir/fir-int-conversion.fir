// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck --check-prefixes=COMMON,DEFAULT %s
// RUN: fir-opt --kind-mapping="i1:4,i2:8,i4:16,i8:32,i16:64" --split-input-file --fir-to-llvm-ir %s | FileCheck --check-prefixes=COMMON,ALL-CUSTOM %s
// RUN: fir-opt --kind-mapping="i2:1,i4:8,i16:32" --split-input-file --fir-to-llvm-ir %s | FileCheck --check-prefixes=COMMON,SOME-CUSTOM %s

// Test `!fir.integer<KIND>` conversion with and without kind-mapping string

func.func private @foo0(%arg0: !fir.int<1>)
// COMMON-LABEL: foo0
// DEFAULT-SAME: i8
// ALL-CUSTOM-SAME: i4
// SOME-CUSTOM-SAME: i8

func.func private @foo1(%arg0: !fir.int<2>)
// COMMON-LABEL: foo1
// DEFAULT-SAME: i16
// ALL-CUSTOM-SAME: i8
// SOME-CUSTOM-SAME: i1

func.func private @foo2(%arg0: !fir.int<4>)
// COMMON-LABEL: foo2
// DEFAULT-SAME: i32
// ALL-CUSTOM-SAME: i16
// SOME-CUSTOM-SAME: i8

func.func private @foo3(%arg0: !fir.int<8>)
// COMMON-LABEL: foo3
// DEFAULT-SAME: i64
// ALL-CUSTOM-SAME: i32
// SOME-CUSTOM-SAME: i64

func.func private @foo4(%arg0: !fir.int<16>)
// COMMON-LABEL: foo4
// DEFAULT-SAME: i128
// ALL-CUSTOM-SAME: i64
// SOME-CUSTOM-SAME: i32
