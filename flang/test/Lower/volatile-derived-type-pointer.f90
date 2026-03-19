! RUN: bbc %s -o - --strict-fir-volatile-verifier | FileCheck %s

! Ensure that assignments between volatile classes/derived type pointer/targets
! lower to the correct hlfir declare/designate operations.

module m
  type :: dt
    character :: c0="!"
    integer   :: i=0
    character :: c1="!"
  end type
  end module
  program dataptrvolatile
  use m
  implicit none
  type(dt),  volatile , target  :: arr(100, 100), arr1(10000), t(100,100)
  class(dt), volatile , pointer :: ptr(:, :)
  integer             :: i, j
  do i =1, 100
  do j =i, 100
    arr(i:, j:) = dt(i=-i)
    ptr(i:, j:) => arr(i:, j:)
    t(i:, j:) = ptr(i:, j:)
  end do
  end do
end

! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, !fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEarr1"} : (!fir.ref<!fir.array<10000x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10000x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, !fir.ref<!fir.array<10000x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEt"} : (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, !fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>) -> (!fir.ref<!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, !fir.ref<!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0{"c0"}   typeparams %{{.+}} : (!fir.ref<!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX21"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0{"i"}   : (!fir.ref<!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>) -> !fir.ref<i32>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0{"c1"}   typeparams %{{.+}} : (!fir.ref<!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX21"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0 (%{{.+}}:%{{.+}}:%{{.+}}, %{{.+}}:%{{.+}}:%{{.+}})  shape %{{.+}} : (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0 (%{{.+}}:%{{.+}}:%{{.+}}, %{{.+}}:%{{.+}}:%{{.+}})  shape %{{.+}} : (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}} (%{{.+}}:%{{.+}}:%{{.+}}, %{{.+}}:%{{.+}}:%{{.+}})  shape %{{.+}} : (!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0 (%{{.+}}:%{{.+}}:%{{.+}}, %{{.+}}:%{{.+}}:%{{.+}})  shape %{{.+}} : (!fir.ref<!fir.array<100x100x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTdt{c0:!fir.char<1>,i:i32,c1:!fir.char<1>}>>, volatile>
