! RUN: bbc -polymorphic-type -emit-fir -hlfir=false -o - %s | FileCheck %s

module m
  type t
    integer n
  end type
  interface write(formatted)
    module procedure wft
  end interface
 contains
  ! CHECK-LABEL: @_QMmPwft
  subroutine wft(dtv, unit, iotype, v_list, iostat, iomsg)
    class(t), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    iostat = 0
    write(unit,*,iostat=iostat,iomsg=iomsg) 'wft was called: ', dtv%n
  end subroutine

  ! CHECK-LABEL: @_QMmPwftd
  subroutine wftd(dtv, unit, iotype, v_list, iostat, iomsg)
    type(t), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    iostat = 0
    write(unit,*,iostat=iostat,iomsg=iomsg) 'wftd: ', dtv%n
  end subroutine

  ! CHECK-LABEL: @_QMmPtest1
  subroutine test1
    import, all
    ! CHECK:   %[[V_14:[0-9]+]] = fir.field_index n, !fir.type<_QMmTt{n:i32}>
    ! CHECK:   %[[V_15:[0-9]+]] = fir.coordinate_of %{{.*}}, %[[V_14]] : (!fir.ref<!fir.type<_QMmTt{n:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK:   fir.store %c1{{.*}} to %[[V_15]] : !fir.ref<i32>
    ! CHECK:   %[[V_16:[0-9]+]] = fir.embox %{{.*}} : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<!fir.type<_QMmTt{n:i32}>>
    ! CHECK:   %[[V_17:[0-9]+]] = fir.convert %[[V_16]] : (!fir.box<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<none>
    ! CHECK:   %[[V_18:[0-9]+]] = fir.address_of(@_QQMmFtest1.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
    ! CHECK:   %[[V_19:[0-9]+]] = fir.convert %[[V_18]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
    ! CHECK:   %[[V_20:[0-9]+]] = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %[[V_17]], %[[V_19]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
    print *, 'test1 outer, should call wft: ', t(1)
    block
      import, only: t
      ! CHECK:   %[[V_35:[0-9]+]] = fir.field_index n, !fir.type<_QMmTt{n:i32}>
      ! CHECK:   %[[V_36:[0-9]+]] = fir.coordinate_of %{{.*}}, %[[V_35]] : (!fir.ref<!fir.type<_QMmTt{n:i32}>>, !fir.field) -> !fir.ref<i32>
      ! CHECK:   fir.store %c2{{.*}} to %[[V_36]] : !fir.ref<i32>
      ! CHECK:   %[[V_37:[0-9]+]] = fir.embox %{{.*}} : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<!fir.type<_QMmTt{n:i32}>>
      ! CHECK:   %[[V_38:[0-9]+]] = fir.convert %[[V_37]] : (!fir.box<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<none>
      ! CHECK:   %[[V_39:[0-9]+]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
      ! CHECK:   %[[V_40:[0-9]+]] = fir.convert %[[V_39]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
      ! CHECK:   %[[V_41:[0-9]+]] = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %[[V_38]], %[[V_40]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
      print *, 'test1 block, should not call wft: ', t(2)
    end block
  end subroutine

  ! CHECK-LABEL: @_QMmPtest2
  subroutine test2
    ! CHECK:   %[[V_13:[0-9]+]] = fir.field_index n, !fir.type<_QMmTt{n:i32}>
    ! CHECK:   %[[V_14:[0-9]+]] = fir.coordinate_of %{{.*}}, %[[V_13]] : (!fir.ref<!fir.type<_QMmTt{n:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK:   fir.store %c3{{.*}} to %[[V_14]] : !fir.ref<i32>
    ! CHECK:   %[[V_15:[0-9]+]] = fir.embox %{{.*}} : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<!fir.type<_QMmTt{n:i32}>>
    ! CHECK:   %[[V_16:[0-9]+]] = fir.convert %[[V_15]] : (!fir.box<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<none>
    ! CHECK:   %[[V_17:[0-9]+]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
    ! CHECK:   %[[V_18:[0-9]+]] = fir.convert %[[V_17]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
    ! CHECK:   %[[V_19:[0-9]+]] = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %[[V_16]], %[[V_18]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1

    import, only: t
    print *, 'test2, should not call wft: ', t(3)
  end subroutine

  ! CHECK-LABEL: @_QMmPtest3
  subroutine test3(p, x)
    procedure(wftd) p
    type(t), intent(in) :: x
    interface write(formatted)
      procedure p
    end interface

    ! CHECK:     %[[V_3:[0-9]+]] = fir.embox %arg1 : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<!fir.type<_QMmTt{n:i32}>>
    ! CHECK:     %[[V_4:[0-9]+]] = fir.convert %[[V_3]] : (!fir.box<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<none>
    ! CHECK:     %[[V_5:[0-9]+]] = fir.alloca !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
    ! CHECK:     %[[V_6:[0-9]+]] = fir.undefined !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
    ! CHECK:     %[[V_7:[0-9]+]] = fir.address_of(@_QMmE.dt.t)
    ! CHECK:     %[[V_8:[0-9]+]] = fir.convert %[[V_7]] : {{.*}} -> !fir.ref<none>
    ! CHECK:     %[[V_9:[0-9]+]] = fir.insert_value %[[V_6]], %[[V_8]], [0 : index, 0 : index] : (!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>, !fir.ref<none>) -> !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
    ! CHECK:     %[[V_10:[0-9]+]] = fir.box_addr %arg0 : (!fir.boxproc<() -> ()>) -> !fir.ref<none>
    ! CHECK:     %[[V_11:[0-9]+]] = fir.insert_value %[[V_9]], %[[V_10]], [0 : index, 1 : index] : (!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>, !fir.ref<none>) -> !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
    ! CHECK:     %[[V_12:[0-9]+]] = fir.insert_value %[[V_11]], %c2{{.*}}, [0 : index, 2 : index] : (!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>, i32) -> !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
    ! CHECK:     %[[V_13:[0-9]+]] = fir.insert_value %[[V_12]], %false, [0 : index, 3 : index] : (!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>, i1) -> !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
    ! CHECK:     fir.store %[[V_13]] to %[[V_5]] : !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>
    ! CHECK:     %[[V_14:[0-9]+]] = fir.alloca tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
    ! CHECK:     %[[V_15:[0-9]+]] = fir.undefined tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
    ! CHECK:     %[[V_16:[0-9]+]] = fir.insert_value %[[V_15]], %c1{{.*}}, [0 : index] : (tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>, i64) -> tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
    ! CHECK:     %[[V_17:[0-9]+]] = fir.insert_value %[[V_16]], %[[V_5]], [1 : index] : (tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>) -> tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
    ! CHECK:     %[[V_18:[0-9]+]] = fir.insert_value %[[V_17]], %true, [2 : index] : (tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>, i1) -> tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
    ! CHECK:     fir.store %[[V_18]] to %[[V_14]] : !fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
    ! CHECK:     %[[V_19:[0-9]+]] = fir.convert %[[V_14]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
    ! CHECK:     %[[V_20:[0-9]+]] = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %[[V_4]], %[[V_19]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
    print *, x
  end subroutine
end module

! CHECK-LABEL: @_QQmain
program p
  use m
  character*3 ccc(4)
  namelist /nnn/ jjj, ccc

  ! CHECK:   fir.call @_QMmPtest1
  call test1
  ! CHECK:   fir.call @_QMmPtest2
  call test2
  ! CHECK:   fir.call @_QMmPtest3
  call test3(wftd, t(17))

  ! CHECK:   %[[V_95:[0-9]+]] = fir.field_index n, !fir.type<_QMmTt{n:i32}>
  ! CHECK:   %[[V_96:[0-9]+]] = fir.coordinate_of %{{.*}}, %[[V_95]] : (!fir.ref<!fir.type<_QMmTt{n:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:   fir.store %c4{{.*}} to %[[V_96]] : !fir.ref<i32>
  ! CHECK:   %[[V_97:[0-9]+]] = fir.embox %{{.*}} : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<!fir.type<_QMmTt{n:i32}>>
  ! CHECK:   %[[V_98:[0-9]+]] = fir.convert %[[V_97]] : (!fir.box<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<none>
  ! CHECK:   %[[V_99:[0-9]+]] = fir.address_of(@_QQF.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
  ! CHECK:   %[[V_100:[0-9]+]] = fir.convert %[[V_99]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
  ! CHECK:   %[[V_101:[0-9]+]] = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %[[V_98]], %[[V_100]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
  print *, 'main, should call wft: ', t(4)
end

! CHECK: fir.global linkonce @_QQMmFtest1.nonTbpDefinedIoTable.list constant : !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
! CHECK: fir.global linkonce @_QQMmFtest1.nonTbpDefinedIoTable constant : tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
! CHECK: fir.global linkonce @default.nonTbpDefinedIoTable constant : tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
! CHECK: fir.global linkonce @_QQF.nonTbpDefinedIoTable.list constant : !fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>
! CHECK: fir.global linkonce @_QQF.nonTbpDefinedIoTable constant : tuple<i64, !fir.ref<!fir.array<1xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
