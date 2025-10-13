// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// Test fir.call modref with internal procedures after the host function has been inlined in
// some other function. This checks that the last hlfir.declare "internal_assoc" flags that
// marks a variable that was captured is still considered even though there is no such flags
// on the declare at the top of the chain.
//
// In other words, in the following Fortran example, "x" should be considered
// modified by "call internal_proc" after "call inline_me" was inlined into
// "test".
//
//    subroutine test()
//      real :: x(10)
//      call inline_me(x)
//    end subroutine
//    
//    subroutine inline_me(x)
//      real :: x(10)
//      call internal_proc()
//    contains
//      subroutine internal_proc()
//        call some_external(x)
//      end subroutine
//    end subroutine

func.func @_QPtest() {                                                                 
  %c0_i32 = arith.constant 0 : i32                                                     
  %c10 = arith.constant 10 : index                                                     
  %0 = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtestEx"}       
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>                                       
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = fir.dummy_scope : !fir.dscope
  %4:2 = hlfir.declare %2#1(%1) dummy_scope %3 {test.ptr = "x", fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFinline_meEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %5 = fir.alloca tuple<!fir.box<!fir.array<10xf32>>>
  %6 = fir.coordinate_of %5, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xf32>>>
  %7 = fir.embox %4#1(%1) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  fir.store %7 to %6 : !fir.ref<!fir.box<!fir.array<10xf32>>>                          
  fir.call @_QFinline_mePinternal_proc(%5) {test.ptr="internal_proc"} : (!fir.ref<tuple<!fir.box<!fir.array<10xf32>>>>) -> ()
  return
} 
func.func private @_QFinline_mePinternal_proc(!fir.ref<tuple<!fir.box<!fir.array<10xf32>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QPinline_me}

// CHECK-LABEL: Testing : "_QPtest"
// CHECK: internal_proc -> x#0: ModRef
