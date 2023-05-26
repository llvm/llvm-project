! Test lowering of OPEN statment options
! RUN: bbc %s -emit-fir -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_convert_specifier(
subroutine test_convert_specifier(unit)
  integer :: unit
  ! CHECK: %[[cookie:.*]] = fir.call @_FortranAioBeginOpenUnit(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[be_str:.*]] = fir.address_of(@[[be_str_name:.*]]) : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[len:.*]] = arith.constant 10 : index
  ! CHECK: %[[be_str_conv:.*]] = fir.convert %[[be_str]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[len_conv:.*]] = fir.convert %[[len]] : (index) -> i64
  ! CHECK: %{{.*}} = fir.call @_FortranAioSetConvert(%[[cookie]], %[[be_str_conv]], %[[len_conv]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
  ! CHECK: %{{.*}} = fir.call @_FortranAioEndIoStatement(%[[cookie]]) {{.*}}: (!fir.ref<i8>) -> i32
  open(unit, form="UNFORMATTED", convert="BIG_ENDIAN")
  close(unit)
end subroutine

! CHECK: fir.global internal @[[be_str_name]] constant : !fir.char<1,10> {
! CHECK: %[[be_str_lit:.*]] = fir.string_lit "BIG_ENDIAN"(10) : !fir.char<1,10>
! CHECK: fir.has_value %[[be_str_lit]] : !fir.char<1,10>
