!RUN: bbc -emit-fir -o - %s 2>&1 | FileCheck %s
module m
  type samename
    sequence
    integer n
  end type
  interface samename
  end interface
  interface write(formatted)
    module procedure :: write_formatted
  end interface
 contains
  subroutine write_formatted(t, unit, iotype, v_list, iostat, iomsg)
    type(samename), intent(in) :: t
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end
  subroutine test
    print *, samename(123)
  end
end

!CHECK: %[[VAL_8:.*]] = fir.address_of(@_QQMmFtest.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>>
!CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>>) -> !fir.ref<none>
!CHECK: %{{.*}} = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %{{.*}}, %[[VAL_9]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
