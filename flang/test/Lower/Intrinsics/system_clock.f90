! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPsystem_clock_test() {
subroutine system_clock_test()
  integer(4) :: c
  integer(8) :: m
  real :: r
  ! CHECK-DAG: %[[c_addr:.*]] = fir.alloca i32 {bindc_name = "c"
  ! CHECK-DAG: %[[c:.*]]:2 = hlfir.declare %[[c_addr]]
  ! CHECK-DAG: %[[m_addr:.*]] = fir.alloca i64 {bindc_name = "m"
  ! CHECK-DAG: %[[m:.*]]:2 = hlfir.declare %[[m_addr]]
  ! CHECK-DAG: %[[r_addr:.*]] = fir.alloca f32 {bindc_name = "r"
  ! CHECK-DAG: %[[r:.*]]:2 = hlfir.declare %[[r_addr]]
  ! CHECK: %[[c4:.*]] = arith.constant 4 : i32
  ! CHECK: %[[Count:.*]] = fir.call @_FortranASystemClockCount(%[[c4]]) {{.*}}: (i32) -> i64
  ! CHECK: %[[Count1:.*]] = fir.convert %[[Count]] : (i64) -> i32
  ! CHECK: fir.store %[[Count1]] to %[[c]]#0 : !fir.ref<i32>
  ! CHECK: %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8]]) {{.*}}: (i32) -> i64
  ! CHECK: %[[Rate1:.*]] = fir.convert %[[Rate]] : (i64) -> f32
  ! CHECK: fir.store %[[Rate1]] to %[[r]]#0 : !fir.ref<f32>
  ! CHECK: %[[c8_2:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Max:.*]] = fir.call @_FortranASystemClockCountMax(%[[c8_2]]) {{.*}}: (i32) -> i64
  ! CHECK: fir.store %[[Max]] to %[[m]]#0 : !fir.ref<i64>
  call system_clock(c, r, m)
! print*, c, r, m
  ! CHECK-NOT: fir.call
  ! CHECK: %[[c8_3:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8_3]]) {{.*}}: (i32) -> i64
  ! CHECK: fir.store %[[Rate]] to %[[m]]#0 : !fir.ref<i64>
  call system_clock(count_rate=m)
  ! CHECK-NOT: fir.call
! print*, m
end subroutine

! CHECK-LABEL: func.func @_QPss(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i64> {fir.bindc_name = "count", fir.optional})
subroutine ss(count)
  ! CHECK: %[[count:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[count_max_box:.*]] = fir.alloca !fir.box<!fir.heap<i64>> {bindc_name = "count_max"
  ! CHECK: %[[count_max:.*]]:2 = hlfir.declare %[[count_max_box]]
  ! CHECK: %[[count_rate_box:.*]] = fir.alloca !fir.box<!fir.ptr<i64>> {bindc_name = "count_rate"
  ! CHECK: %[[count_rate:.*]]:2 = hlfir.declare %[[count_rate_box]]
  ! CHECK: %[[count_rate_base:.*]] = fir.alloca i64 {bindc_name = "count_rate_", fir.target
  ! CHECK: %[[count_rate_base_decl:.*]]:2 = hlfir.declare %[[count_rate_base]]
  ! CHECK: %[[embox:.*]] = fir.embox %[[count_rate_base_decl]]#0 : (!fir.ref<i64>) -> !fir.box<!fir.ptr<i64>>
  ! CHECK: fir.store %[[embox]] to %[[count_rate]]#0 : !fir.ref<!fir.box<!fir.ptr<i64>>>
  ! CHECK: %[[alloc:.*]] = fir.allocmem i64
  ! CHECK: %[[embox_max:.*]] = fir.embox %[[alloc]] : (!fir.heap<i64>) -> !fir.box<!fir.heap<i64>>
  ! CHECK: fir.store %[[embox_max]] to %[[count_max]]#0 : !fir.ref<!fir.box<!fir.heap<i64>>>
  ! CHECK: %[[box_rate:.*]] = fir.load %[[count_rate]]#0 : !fir.ref<!fir.box<!fir.ptr<i64>>>
  ! CHECK: %[[addr_rate:.*]] = fir.box_addr %[[box_rate]] : (!fir.box<!fir.ptr<i64>>) -> !fir.ptr<i64>
  ! CHECK: %[[box_max_val:.*]] = fir.load %[[count_max]]#0 : !fir.ref<!fir.box<!fir.heap<i64>>>
  ! CHECK: %[[addr_max:.*]] = fir.box_addr %[[box_max_val]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
  ! CHECK: %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[val_count:.*]] = fir.call @_FortranASystemClockCount(%[[c8]]) {{.*}} : (i32) -> i64
  ! CHECK: fir.store %[[val_count]] to %[[count]]#0 : !fir.ref<i64>
  ! CHECK: %[[addr_rate_i64:.*]] = fir.convert %[[addr_rate]] : (!fir.ptr<i64>) -> i64
  ! CHECK: %[[c0_i64:.*]] = arith.constant 0 : i64
  ! CHECK: %[[is_rate_present:.*]] = arith.cmpi ne, %[[addr_rate_i64]], %[[c0_i64]] : i64
  ! CHECK: fir.if %[[is_rate_present]] {
  ! CHECK:   %[[c8_new:.*]] = arith.constant 8 : i32
  ! CHECK:   %[[val_rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8_new]]) {{.*}} : (i32) -> i64
  ! CHECK:   fir.store %[[val_rate]] to %[[addr_rate]] : !fir.ptr<i64>
  ! CHECK: }
  ! CHECK: %[[addr_max_i64:.*]] = fir.convert %[[addr_max]] : (!fir.heap<i64>) -> i64
  ! CHECK: %[[c0_i64_2:.*]] = arith.constant 0 : i64
  ! CHECK: %[[is_max_present:.*]] = arith.cmpi ne, %[[addr_max_i64]], %[[c0_i64_2]] : i64
  ! CHECK: fir.if %[[is_max_present]] {
  ! CHECK:   %[[c8_new2:.*]] = arith.constant 8 : i32
  ! CHECK:   %[[val_max:.*]] = fir.call @_FortranASystemClockCountMax(%[[c8_new2]]) {{.*}} : (i32) -> i64
  ! CHECK:   fir.store %[[val_max]] to %[[addr_max]] : !fir.heap<i64>
  ! CHECK: }

  ! CHECK: %[[is_count_present:.*]] = fir.is_present %[[count]]#0 : (!fir.ref<i64>) -> i1
  ! CHECK: fir.if %[[is_count_present]] {
  ! CHECK:   %[[io_begin:.*]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:   %[[val_count_io:.*]] = fir.load %[[count]]#0 : !fir.ref<i64>
  ! CHECK:   fir.call @_FortranAioOutputInteger64(%[[io_begin]], %[[val_count_io]])
  ! CHECK: } else {
  ! CHECK:   fir.if %{{.*}} {
  ! CHECK:     fir.call @_FortranASystemClockCountRate
  ! CHECK:   }
  ! CHECK:   fir.if %{{.*}} {
  ! CHECK:     fir.call @_FortranASystemClockCountMax
  ! CHECK:   }
  ! CHECK: }

  ! CHECK: %[[is_count_present_2:.*]] = fir.is_present %[[count]]#0 : (!fir.ref<i64>) -> i1
  ! CHECK: fir.if %[[is_count_present_2]] {
  ! CHECK:   %[[c0_i64_3:.*]] = arith.constant 0 : i64
  ! CHECK:   hlfir.assign %[[c0_i64_3]] to %[[count]]#0 : i64, !fir.ref<i64>
  ! CHECK: }

  ! CHECK: %[[zero_ptr:.*]] = fir.zero_bits !fir.ptr<i64>
  ! CHECK: %[[embox_null:.*]] = fir.embox %[[zero_ptr]]
  ! CHECK: fir.store %[[embox_null]] to %[[count_rate]]#0
  ! CHECK: %[[box_max_free:.*]] = fir.load %[[count_max]]#0
  ! CHECK: %[[addr_max_free:.*]] = fir.box_addr %[[box_max_free]]
  ! CHECK: fir.freemem %[[addr_max_free]]
  ! CHECK: %[[zero_heap:.*]] = fir.zero_bits !fir.heap<i64>
  ! CHECK: %[[embox_null_max:.*]] = fir.embox %[[zero_heap]]
  ! CHECK: fir.store %[[embox_null_max]] to %[[count_max]]#0

  ! CHECK: %[[val_count_last:.*]] = fir.call @_FortranASystemClockCount
  ! CHECK: fir.store %[[val_count_last]] to %[[count]]#0
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.call @_FortranASystemClockCountRate
  ! CHECK: }
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.call @_FortranASystemClockCountMax
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
