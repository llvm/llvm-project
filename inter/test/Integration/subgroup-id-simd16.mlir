// REQUIRES: host-supports-inter-bmg
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
// RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
// RUN: inter-runner --group-size 256 %t.bin subgroup_id_simd16 256 out | %python %S/../../verify.py '(0 if i%16 == 0 else 0xA5A5A5A5)'

module {
  func.func @subgroup_id_simd16(%out: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.required_work_group_size = [256 : i32, 1 : i32, 1 : i32],
      xw.simd_width = 16 : i32} {
    %id = xw.subgroup_id : i32
    %id64 = xw.cast intconvert %id policy {extension = #xw.cast_extension<zero>}
        : i32 -> i64
    %four = xw.constant 4 : i64
    %column = xw.binary remui %id64, %four : i64, i64 -> i64
    %row = xw.binary divui %id64, %four : i64, i64 -> i64
    %row4 = xw.binary muli %row, %four : i64, i64 -> i64
    %linear = xw.binary addi %row4, %column : i64, i64 -> i64
    %six = xw.constant 6 : i64
    %offset = xw.binary shli %linear, %six : i64, i64 -> i64
    %offsets = xw.splat %offset : i64 -> !xw.simd<i64, 16>
    %addresses = xw.ptradd %out, %offsets
        : !xw.ptr<#xw.global>, !xw.simd<i64, 16>
          -> !xw.simd<!xw.ptr<#xw.global>, 16>
    %values = xw.constant 0 : i32 -> !xw.simd<i32, 16>
    %root = xw.token : !xw.mem.token
    %stored = xw.store %values -> %addresses after %root
        : (!xw.simd<i32, 16>, !xw.simd<!xw.ptr<#xw.global>, 16>,
           !xw.mem.token) -> !xw.mem.token
    return
  }
}
