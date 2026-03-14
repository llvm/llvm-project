! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

  ! CHECK-LABEL: func @_QQmain
  ! CHECK: %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  ! CHECK: fir.do_loop
  do i=-20,20
    ! CHECK: %[[V_5:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK: %[[V_6:[0-9]+]] = arith.cmpi eq, %[[V_5]], %c-1{{.*}} : i32
    ! CHECK: fir.if %[[V_6]] {
    ! CHECK:   %[[V_20:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK:   %[[V_21:[0-9]+]] = fir.call @_FortranAioOutputInteger32(%{{.*}} %[[V_20]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
    ! CHECK: }
    if (is_iostat_end(i)) print*, "iostat_end =", i
    ! CHECK: %[[V_7:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK: %[[V_8:[0-9]+]] = arith.cmpi eq, %[[V_7]], %c-2{{.*}} : i32
    ! CHECK: fir.if %[[V_8]] {
    ! CHECK:   %[[V_20:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK:   %[[V_21:[0-9]+]] = fir.call @_FortranAioOutputInteger32(%{{.*}} %[[V_20]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
    ! CHECK: }
    if (is_iostat_eor(i)) print*, "iostat_eor =", i
  ! CHECK: }
  enddo
end
