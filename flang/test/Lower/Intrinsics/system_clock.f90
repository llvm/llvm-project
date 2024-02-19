! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: system_clock_test
subroutine system_clock_test()
  integer(4) :: c
  integer(8) :: m
  real :: r
  ! CHECK-DAG: %[[c:.*]] = fir.alloca i32 {bindc_name = "c"
  ! CHECK-DAG: %[[m:.*]] = fir.alloca i64 {bindc_name = "m"
  ! CHECK-DAG: %[[r:.*]] = fir.alloca f32 {bindc_name = "r"
  ! CHECK: %[[c4:.*]] = arith.constant 4 : i32
  ! CHECK: %[[Count:.*]] = fir.call @_FortranASystemClockCount(%[[c4]]) {{.*}}: (i32) -> i64
  ! CHECK: %[[Count1:.*]] = fir.convert %[[Count]] : (i64) -> i32
  ! CHECK: fir.store %[[Count1]] to %[[c]] : !fir.ref<i32>
  ! CHECK: %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8]]) {{.*}}: (i32) -> i64
  ! CHECK: %[[Rate1:.*]] = fir.convert %[[Rate]] : (i64) -> f32
  ! CHECK: fir.store %[[Rate1]] to %[[r]] : !fir.ref<f32>
  ! CHECK: %[[c8_2:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Max:.*]] = fir.call @_FortranASystemClockCountMax(%[[c8_2]]) {{.*}}: (i32) -> i64
  ! CHECK: fir.store %[[Max]] to %[[m]] : !fir.ref<i64>
  call system_clock(c, r, m)
! print*, c, r, m
  ! CHECK-NOT: fir.call
  ! CHECK: %[[c8_3:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8_3]]) {{.*}}: (i32) -> i64
  ! CHECK: fir.store %[[Rate]] to %[[m]] : !fir.ref<i64>
  call system_clock(count_rate=m)
  ! CHECK-NOT: fir.call
! print*, m
end subroutine

! CHECK-LABEL: @_QPss
subroutine ss(count)
  ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca !fir.box<!fir.heap<i64>> {bindc_name = "count_max", uniq_name = "_QFssEcount_max"}
  ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca !fir.heap<i64> {uniq_name = "_QFssEcount_max.addr"}
  ! CHECK:   %[[V_2:[0-9]+]] = fir.zero_bits !fir.heap<i64>
  ! CHECK:   fir.store %[[V_2]] to %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:   %[[V_3:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<i64>> {bindc_name = "count_rate", uniq_name = "_QFssEcount_rate"}
  ! CHECK:   %[[V_4:[0-9]+]] = fir.alloca !fir.ptr<i64> {uniq_name = "_QFssEcount_rate.addr"}
  ! CHECK:   %[[V_5:[0-9]+]] = fir.zero_bits !fir.ptr<i64>
  ! CHECK:   fir.store %[[V_5]] to %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.alloca i64 {bindc_name = "count_rate_", fir.target, uniq_name = "_QFssEcount_rate_"}
  ! CHECK:   %[[V_7:[0-9]+]] = fir.convert %[[V_6]] : (!fir.ref<i64>) -> !fir.ptr<i64>
  ! CHECK:   fir.store %[[V_7]] to %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:   %[[V_8:[0-9]+]] = fir.allocmem i64 {fir.must_be_heap = true, uniq_name = "_QFssEcount_max.alloc"}
  ! CHECK:   fir.store %[[V_8]] to %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:   %[[V_9:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:   %[[V_10:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:   %[[V_11:[0-9]+]] = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  ! CHECK:   fir.if %[[V_11]] {
  ! CHECK:     %[[V_29:[0-9]+]] = fir.call @_FortranASystemClockCount(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:     fir.store %[[V_29]] to %arg0 : !fir.ref<i64>
  ! CHECK:   }
  ! CHECK:   %[[V_12:[0-9]+]] = fir.convert %[[V_9]] : (!fir.ptr<i64>) -> i64
  ! CHECK:   %[[V_13:[0-9]+]] = arith.cmpi ne, %[[V_12]], %c0{{.*}} : i64
  ! CHECK:   fir.if %[[V_13]] {
  ! CHECK:     %[[V_29:[0-9]+]] = fir.call @_FortranASystemClockCountRate(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:     fir.store %[[V_29]] to %[[V_9]] : !fir.ptr<i64>
  ! CHECK:   }
  ! CHECK:   %[[V_14:[0-9]+]] = fir.convert %[[V_10]] : (!fir.heap<i64>) -> i64
  ! CHECK:   %[[V_15:[0-9]+]] = arith.cmpi ne, %[[V_14]], %c0{{.*}} : i64
  ! CHECK:   fir.if %[[V_15]] {
  ! CHECK:     %[[V_29]] = fir.call @_FortranASystemClockCountMax(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:     fir.store %[[V_29]] to %[[V_10]] : !fir.heap<i64>
  ! CHECK:   }
  ! CHECK:   %[[V_16:[0-9]+]] = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  ! CHECK:   fir.if %[[V_16]] {
  ! CHECK:     %[[V_31:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_32:[0-9]+]] = fir.load %arg0 : !fir.ref<i64>
  ! CHECK:     %[[V_33:[0-9]+]] = fir.call @_FortranAioOutputInteger64(%[[V_31]], %[[V_32]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK:     %[[V_34:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:     %[[V_35:[0-9]+]] = fir.load %[[V_34]] : !fir.ptr<i64>
  ! CHECK:     %[[V_36:[0-9]+]] = fir.call @_FortranAioOutputInteger64(%[[V_31]], %[[V_35]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK:     %[[V_37:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:     %[[V_38:[0-9]+]] = fir.load %[[V_37]] : !fir.heap<i64>
  ! CHECK:     %[[V_39:[0-9]+]] = fir.call @_FortranAioOutputInteger64(%[[V_31]], %[[V_38]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK:     %[[V_40:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_31]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK:   } else {
  ! CHECK:     %[[V_29:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:     %[[V_30:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:     %[[V_31:[0-9]+]] = fir.convert %[[V_29]] : (!fir.ptr<i64>) -> i64
  ! CHECK:     %[[V_32:[0-9]+]] = arith.cmpi ne, %[[V_31]], %c0{{.*}} : i64
  ! CHECK:     fir.if %[[V_32]] {
  ! CHECK:       %[[V_45:[0-9]+]] = fir.call @_FortranASystemClockCountRate(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:       fir.store %[[V_45]] to %[[V_29]] : !fir.ptr<i64>
  ! CHECK:     }
  ! CHECK:     %[[V_33:[0-9]+]] = fir.convert %[[V_30]] : (!fir.heap<i64>) -> i64
  ! CHECK:     %[[V_34:[0-9]+]] = arith.cmpi ne, %[[V_33]], %c0{{.*}} : i64
  ! CHECK:     fir.if %[[V_34]] {
  ! CHECK:       %[[V_45]] = fir.call @_FortranASystemClockCountMax(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:       fir.store %[[V_45]] to %[[V_30]] : !fir.heap<i64>
  ! CHECK:     }
  ! CHECK:     %[[V_37:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_38:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:     %[[V_39:[0-9]+]] = fir.load %[[V_38]] : !fir.ptr<i64>
  ! CHECK:     %[[V_40:[0-9]+]] = fir.call @_FortranAioOutputInteger64(%[[V_37]], %[[V_39]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK:     %[[V_41:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:     %[[V_42:[0-9]+]] = fir.load %[[V_41]] : !fir.heap<i64>
  ! CHECK:     %[[V_43:[0-9]+]] = fir.call @_FortranAioOutputInteger64(%[[V_37]], %[[V_42]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK:     %[[V_44:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_37]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK:   }
  ! CHECK:   %[[V_17:[0-9]+]] = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  ! CHECK:   fir.if %[[V_17]] {
  ! CHECK:     %[[C_0:c[0-9a-z_]+]] = arith.constant 0 : i64
  ! CHECK:     fir.store %[[C_0]] to %arg0 : !fir.ref<i64>
  ! CHECK:   } else {
  ! CHECK:   }
  ! CHECK:   %[[V_18:[0-9]+]] = fir.zero_bits !fir.ptr<i64>
  ! CHECK:   fir.store %[[V_18]] to %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:   %[[V_19:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:   fir.freemem %[[V_19]] : !fir.heap<i64>
  ! CHECK:   %[[V_20:[0-9]+]] = fir.zero_bits !fir.heap<i64>
  ! CHECK:   fir.store %[[V_20]] to %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:   %[[V_21:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<!fir.ptr<i64>>
  ! CHECK:   %[[V_22:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.heap<i64>>
  ! CHECK:   %[[V_23:[0-9]+]] = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  ! CHECK:   fir.if %[[V_23]] {
  ! CHECK:     %[[V_29]] = fir.call @_FortranASystemClockCount(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:     fir.store %[[V_29]] to %arg0 : !fir.ref<i64>
  ! CHECK:   }
  ! CHECK:   %[[V_24:[0-9]+]] = fir.convert %[[V_21]] : (!fir.ptr<i64>) -> i64
  ! CHECK:   %[[V_25:[0-9]+]] = arith.cmpi ne, %[[V_24]], %c0{{.*}} : i64
  ! CHECK:   fir.if %[[V_25]] {
  ! CHECK:     %[[V_29]] = fir.call @_FortranASystemClockCountRate(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:     fir.store %[[V_29]] to %[[V_21]] : !fir.ptr<i64>
  ! CHECK:   }
  ! CHECK:   %[[V_26:[0-9]+]] = fir.convert %[[V_22]] : (!fir.heap<i64>) -> i64
  ! CHECK:   %[[V_27:[0-9]+]] = arith.cmpi ne, %[[V_26]], %c0{{.*}} : i64
  ! CHECK:   fir.if %[[V_27]] {
  ! CHECK:     %[[V_29]] = fir.call @_FortranASystemClockCountMax(%c8{{.*}}_i32) {{.*}}: (i32) -> i64
  ! CHECK:     fir.store %[[V_29]] to %[[V_22]] : !fir.heap<i64>
  ! CHECK:   }
  ! CHECK:   %[[V_28:[0-9]+]] = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  ! CHECK:   fir.if %[[V_28]] {
  ! CHECK:     %[[V_31]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_32]] = fir.load %arg0 : !fir.ref<i64>
  ! CHECK:     %[[V_33]] = fir.call @_FortranAioOutputInteger64(%[[V_31]], %[[V_32]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  ! CHECK:     %[[V_34]] = fir.call @_FortranAioEndIoStatement(%[[V_31]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK:   } else {
  ! CHECK:   }
  ! CHECK:   return
  ! CHECK: }

  integer(8), optional :: count
  integer(8), target :: count_rate_
  integer(8), pointer :: count_rate
  integer(8), allocatable :: count_max

  count_rate => count_rate_
  allocate(count_max)
  call system_clock(count, count_rate, count_max)
  if (present(count)) then
    print*, count, count_rate, count_max
  else
    call system_clock(count_rate=count_rate, count_max=count_max)
    print*, count_rate, count_max
  endif

  if (present(count)) count = 0
  count_rate => null()
  deallocate(count_max)
  call system_clock(count, count_rate, count_max)
  if (present(count)) print*, count
end
