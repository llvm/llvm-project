! This test checks that !$acc routine(name1, name2) and (name1, name2, name3)
! each produce one acc.routine op per named routine, equivalent to separate
! ROUTINE directives.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_multi_routines

! CHECK-DAG: acc.routine @[[r_seq2:.*]] func(@_QMacc_multi_routinesPseq2) seq
! CHECK-DAG: acc.routine @[[r_seq1:.*]] func(@_QMacc_multi_routinesPseq1) seq

  !$acc routine(seq1, seq2) seq

! CHECK-DAG: acc.routine @[[r_gang3:.*]] func(@_QMacc_multi_routinesPgang3) gang
! CHECK-DAG: acc.routine @[[r_gang2:.*]] func(@_QMacc_multi_routinesPgang2) gang
! CHECK-DAG: acc.routine @[[r_gang1:.*]] func(@_QMacc_multi_routinesPgang1) gang

  !$acc routine(gang1, gang2, gang3) gang

contains

  subroutine seq1()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_multi_routinesPseq1()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r_seq1]]]>}

  subroutine seq2()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_multi_routinesPseq2()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r_seq2]]]>}

  subroutine gang1()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_multi_routinesPgang1()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r_gang1]]]>}

  subroutine gang2()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_multi_routinesPgang2()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r_gang2]]]>}

  subroutine gang3()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_multi_routinesPgang3()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r_gang3]]]>}

end module
