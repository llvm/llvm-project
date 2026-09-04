// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @metadata(%input: !xw.ptr<#xw.global>,
                      %output: !xw.ptr<#xw.global>) attributes {
      xw.kernel,
      xw.kernel_args = [
        {access = "read_only", address_space = 1 : i32, alignment = 16 : i64,
         kind = "pointer", offset = 32 : i64, size = 8 : i64},
        {access = "write_only", address_space = 1 : i32, alignment = 16 : i64,
         kind = "pointer", offset = 48 : i64, size = 8 : i64}
      ],
      xw.simd_width = 8 : i32} {
    %local = xw.local_id 0 : !xw.simd<i32, 8>
    return
  }
}

// CHECK-LABEL: func.func @metadata
// CHECK-SAME: #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 16, offset = 32>
// CHECK-SAME: #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 16, offset = 48>
// CHECK: xemachine.mov {{.*}}src0Type = i16
