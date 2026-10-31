// RUN: not inter-opt --inter-select-to-machine="chip=xe3" %s 2>&1 | FileCheck %s --check-prefix=CHIP
// RUN: not inter-opt --inter-select-to-machine="target-features=future" %s 2>&1 | FileCheck %s --check-prefix=FEATURE

// CHIP: unsupported Intel GPU target chip 'xe3'
// FEATURE: unsupported Intel GPU target feature 'future' for chip 'bmg'

module {
  func.func @kernel() attributes {xw.kernel, xw.simd_width = 16 : i32} {
    return
  }
}
