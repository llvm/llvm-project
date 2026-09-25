// RUN: not inter-opt %s --inter-infer-memory-tokens 2>&1 | FileCheck %s

func.func @unsupported_region(%ptr: !xw.ptr<#xw.global>) attributes {
    xw.simd_width = 16 : i32} {
  scf.execute_region {
    %value, %token = xw.load %ptr
        : (!xw.ptr<#xw.global>) -> (!xw.simd<i32, 16>, !xw.mem.token)
    scf.yield
  }
  return
}

// CHECK: error: 'xw.load' op is nested in an unsupported region holder 'scf.execute_region'
