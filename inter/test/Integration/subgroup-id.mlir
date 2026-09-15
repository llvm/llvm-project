// REQUIRES: host-supports-inter-bmg
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
// RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
// RUN: inter-runner --group-size 256 %t.bin subgroup_id 256 out | %python %S/../../verify.py 'i//32'

module {
  func.func @subgroup_id(%out: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.required_work_group_size = [256 : i32, 1 : i32, 1 : i32],
      xw.simd_width = 32 : i32} {
    %id = xw.subgroup_id : i32
    %values = xw.splat %id : i32 -> !xw.simd<i32, 32>
    %gid = xw.global_id 0 : !xw.simd<i64, 32>
    %two = xw.constant 2 : i64 -> !xw.simd<i64, 32>
    %offset = xw.binary shli %gid, %two
        : !xw.simd<i64, 32>, !xw.simd<i64, 32> -> !xw.simd<i64, 32>
    %address = xw.ptradd %out, %offset
        : !xw.ptr<#xw.global>, !xw.simd<i64, 32>
          -> !xw.simd<!xw.ptr<#xw.global>, 32>
    %root = xw.token : !xw.mem.token
    %stored = xw.store %values -> %address after %root
        : (!xw.simd<i32, 32>, !xw.simd<!xw.ptr<#xw.global>, 32>,
           !xw.mem.token) -> !xw.mem.token
    return
  }
}
