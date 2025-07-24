// Test fir.do_concurrent.loop operation with array of symbol reference attributes
// This test demonstrates operations that have ArrayAttr containing SymbolRefAttr elements

// RUN: fir-opt %s --external-name-interop | fir-opt | FileCheck %s

// Define reduction operations that will be referenced in the symbol array
func.func @_QPadd_reduction_i32_init(%arg0: i32, %arg1: !fir.ref<i32>) {
  %0 = arith.constant 0 : i32
  fir.store %0 to %arg1 : !fir.ref<i32>
  return
}

func.func @_QPadd_reduction_i32_combiner(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// Define a local privatizer that will be referenced in local_syms
func.func @_QPlocal_var_privatizer(%arg0: !fir.ref<i32>) -> !fir.ref<i32> {
  return %arg0 : !fir.ref<i32>
}

// Test function demonstrating both local_syms and reduce_syms arrays
func.func @_QPtest_symbol_arrays(%i_lb: index, %i_ub: index, %i_st: index) {
  %local_var = fir.alloca i32
  %sum = fir.alloca i32
  
  fir.do_concurrent {
    %i = fir.alloca i32
    fir.do_concurrent.loop (%i_iv) = (%i_lb) to (%i_ub) step (%i_st)
      local(@_QPlocal_var_privatizer %local_var -> %local_arg : !fir.ref<i32>)
      reduce(@_QPadd_reduction_i32_init #fir.reduce_attr<add> %sum -> %sum_arg : !fir.ref<i32>) {
      %0 = fir.convert %i_iv : (index) -> i32
      fir.store %0 to %i : !fir.ref<i32>
    }
  }
  return
}

// CHECK: local(@local_var_privatizer_
// CHECK: reduce(@add_reduction_i32_init_

