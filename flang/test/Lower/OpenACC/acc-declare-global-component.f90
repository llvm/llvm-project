! Module !$acc declare with an immediate structure component is lowered when
! the whole variable is listed first in the same clause, or on an earlier
! `!$acc declare` line in the module. Each split compiles one module so the
! output must contain exactly one acc.global_ctor and one acc.global_dtor for
! the parent `obj` (no duplicate ctor/dtor for the structure component).

! RUN: split-file %s %t
! RUN: bbc -fopenacc -emit-hlfir %t/mod_same_clause.f90 -o - | FileCheck %s --check-prefixes=SAME
! RUN: bbc -fopenacc -emit-hlfir %t/mod_separate_declare.f90 -o - | FileCheck %s --check-prefixes=SEP

! SAME-DAG: fir.global @_QMmod_same_clauseEobj {acc.declare = #acc.declare<dataClause = acc_create>}
! SAME-COUNT-1: acc.global_ctor @_QMmod_same_clauseEobj_acc_ctor {
! SAME-COUNT-1: acc.global_dtor @_QMmod_same_clauseEobj_acc_dtor {

! SEP-DAG: fir.global @_QMmod_separate_declareEobj {acc.declare = #acc.declare<dataClause = acc_create>}
! SEP-COUNT-1: acc.global_ctor @_QMmod_separate_declareEobj_acc_ctor {
! SEP-COUNT-1: acc.global_dtor @_QMmod_separate_declareEobj_acc_dtor {

//--- mod_same_clause.f90

module mod_same_clause
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array) :: obj
  !$acc declare create(obj, obj%vals)
end module

//--- mod_separate_declare.f90

module mod_separate_declare
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array) :: obj
  !$acc declare create(obj)
  !$acc declare create(obj%vals)
end module
