! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Test non standard definition of a common block as a BIND(C) variable.
! This happens when MPI and MPI_F08 are used inside the same compilation
! unit because MPI uses common blocks while MPI_F08 uses BIND(C) variables
! to refer to the same objects (e.g. mpi_argv_null).

module m_common_var
 character(1) :: var
 common /var_storage/var
end module

module m_bindc_var
  character(1), bind(c, name="var_storage_") :: var
end module

subroutine s1()
  use m_common_var, only : var
  var = "a"
end subroutine

subroutine s2()
  use m_bindc_var, only : var
  print *, var
end subroutine

  call s1()
  call s2()
end

! CHECK: fir.global common @var_storage_(dense<0> : vector<1xi8>) {alignment = 1 : i64} : !fir.array<1xi8>

! CHECK-LABEL: func.func @_QPs1
! CHECK: hlfir.declare %{{.*}} typeparams %c1 storage(%{{.*}}[0]) {uniq_name = "_QMm_common_varEvar"} : (!fir.ref<!fir.char<1>>, index, !fir.ref<!fir.array<1xi8>>) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)

! CHECK-LABEL: func.func @_QPs2
! CHECK: hlfir.declare %{{.*}} typeparams %c1 {fortran_attrs = #fir.var_attrs<bind_c>, uniq_name = "var_storage_"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
