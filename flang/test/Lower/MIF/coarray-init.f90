! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=ALL,COARRAY
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=ALL,NOCOARRAY

program test_init

end

! ALL-LABEL: func.func @main
! COARRAY:   %true = arith.constant true
! COARRAY: fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0, %true) {{.*}} : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>, i1) -> ()

! NOCOARRAY: %false = arith.constant false
! NOCARRAY: fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0, %false) {{.*}} : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>, i1) -> ()

! COARRAY: mif.init -> i32
! NOCOARRAY-NOT: mif.init
