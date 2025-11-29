! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=ALL,COARRAY
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=ALL,NOCOARRAY

program test_init

end

! ALL-LABEL: func.func @main
! ALL: fir.call @_FortranAProgramStart
! COARRAY: mif.init -> i32
! NOCOARRAY-NOT: mif.init

! COARRAY: %[[TRUE:.*]] = arith.constant true
! COARRAY: mif.stop code %[[C0_I32:.*]] quiet %[[TRUE]] : (i32, i1)
! NOCOARRAY-NOT: mif.stop
! NOCOARRAY: fir.call @_FortranAProgramEndStatement
