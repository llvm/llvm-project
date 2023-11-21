! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Ensure that expression rewriting doesn't blow up when many
! mixed complex operations are combined.
! The result of folding (8,1.542956)**9 is checked approximately
! to allow for differing implementations of cpowi.
PROGRAM ProgramName0
 COMPLEX complexVar0
 REAL realVar0
 INTEGER intVar0
 complexVar0 = (8,1.542956)**9/intVar0/realVar0+6+intVar0+intVar0**5-4&
 &+intVar0**intVar0/8**intVar0/intVar&
 &0/intVar0+0/3-3+9/9/5+7+5**0*0**10-0/2**1**2-4+intVar0*intVar0-3**intVar0-6
!CHECK: complexvar0=(-2.{{[0-9]*}}e7_4,1.{{[0-9]*}}e8_4)/(real(intvar0,kind=4),0._4)/(realvar0,0._4)+(6._4,0._4)+(real(intvar0,kind=4),0._4)+(real(intvar0**5_4,kind=4),0._4)-(4._4,0._4)+(real(intvar0**intvar0/8_4**intvar0/intvar0/intvar0,kind=4),0._4)+(0._4,0._4)-(3._4,0._4)+(0._4,0._4)+(7._4,0._4)+(0._4,0._4)-(0._4,0._4)-(4._4,0._4)+(real(intvar0*intvar0,kind=4),0._4)-(real(3_4**intvar0,kind=4),0._4)-(6._4,0._4)
END PROGRAM
