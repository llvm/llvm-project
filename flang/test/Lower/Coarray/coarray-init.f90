! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=ALL,COARRAY
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=ALL,NOCOARRAY

program test_init

end 

! ALL-LABEL: func.func @main
! ALL: fir.call @_FortranAProgramStart
! COARRAY: fir.call @_QMprifPprif_init(%[[ARG:.*]]) fastmath<contract> : (!fir.ref<i32>) -> ()
! NOCOARRAY-NOT: fir.call @_QMprifPprif_init(%[[ARG:.*]]) fastmath<contract> : (!fir.ref<i32>) -> ()
