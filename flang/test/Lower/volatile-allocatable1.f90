! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

! Requires correct propagation of volatility for allocatable nested types.

function allocatable_udt()
  type :: base_type
    integer :: i = 42
  end type
  type, extends(base_type) :: ext_type
    integer :: j = 100
  end type
  integer :: allocatable_udt
  type(ext_type), allocatable, volatile :: v2(:,:)
  allocate(v2(2,3))
  allocatable_udt = v2(1,1)%i
end function
! CHECK-LABEL:   func.func @_QPallocatable_udt() -> i32 {
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.n.i"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.di.base_type.i"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.n.base_type"} : (!fir.ref<!fir.char<1,9>>, index) -> (!fir.ref<!fir.char<1,9>>, !fir.ref<!fir.char<1,9>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.n.j"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.di.ext_type.j"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.n.ext_type"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {uniq_name = "_QFallocatable_udtEallocatable_udt"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFallocatable_udtEv2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.type<{{.*}}>>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.type<{{.*}}>>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.type<{{.*}}>>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.c.base_type"} : (!fir.ref<!fir.array<1x!fir.type<{{.*}}>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<1x!fir.type<{{.*}}>>>, !fir.ref<!fir.array<1x!fir.type<{{.*}}>>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.dt.base_type"} : (!fir.ref<!fir.type<{{.*}}>>) -> (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.dt.ext_type"} : (!fir.ref<!fir.type<{{.*}}>>) -> (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFallocatable_udtE.c.ext_type"} : (!fir.ref<!fir.array<2x!fir.type<{{.*}}>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<2x!fir.type<{{.*}}>>>, !fir.ref<!fir.array<2x!fir.type<{{.*}}>>>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}} (%{{.+}}, %{{.+}})  : (!fir.box<!fir.heap<!fir.array<?x?x!fir.type<{{.*}}>>>, volatile>, index, index) -> !fir.ref<!fir.type<{{.*}}>, volatile>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}{"base_type"}   : (!fir.ref<!fir.type<{{.*}}>, volatile>) -> !fir.ref<!fir.type<{{.*}}>, volatile>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}{"i"}   : (!fir.ref<!fir.type<{{.*}}>, volatile>) -> !fir.ref<i32, volatile>
