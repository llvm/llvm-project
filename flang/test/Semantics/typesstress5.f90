! RUN: %flang_fc1 -fsyntax-only %s 2>&1
! Regression test: deeply nested type extensions with LEN parameters
! must not cause exponential compile time in HasDestruction() or
! IsFinalizable().

module m
  implicit none
  type :: extend1
  end type extend1
  type, extends(extend1) :: extend2(my8int2)
    integer, len :: my8int2
    complex :: mycomplex2
  end type extend2
  type, extends(extend2) :: extend3(my8int3)
    integer, len :: my8int3
    complex :: mycomplex3
  end type extend3
  type, extends(extend3) :: extend4(my8int4)
    integer, len :: my8int4
    complex :: mycomplex4
  end type extend4
  type, extends(extend4) :: extend5(my8int5)
    integer, len :: my8int5
    complex :: mycomplex5
  end type extend5
  type, extends(extend5) :: extend6(my8int6)
    integer, len :: my8int6
    complex :: mycomplex6
  end type extend6
  type, extends(extend6) :: extend7(my8int7)
    integer, len :: my8int7
    complex :: mycomplex7
  end type extend7
  type, extends(extend7) :: extend8(my8int8)
    integer, len :: my8int8
    complex :: mycomplex8
  end type extend8
  type, extends(extend8) :: extend9(my8int9)
    integer, len :: my8int9
    complex :: mycomplex9
  end type extend9
  type, extends(extend9) :: extend10(my8int10)
    integer, len :: my8int10
    complex :: mycomplex10
  end type extend10
  type, extends(extend10) :: extend11(my8int11)
    integer, len :: my8int11
    complex :: mycomplex11
  end type extend11
  type, extends(extend11) :: extend12(my8int12)
    integer, len :: my8int12
    complex :: mycomplex12
  end type extend12
  type, extends(extend12) :: extend13(my8int13)
    integer, len :: my8int13
    complex :: mycomplex13
  end type extend13
  type, extends(extend13) :: extend14(my8int14)
    integer, len :: my8int14
    complex :: mycomplex14
  end type extend14
  type, extends(extend14) :: extend15(my8int15)
    integer, len :: my8int15
    complex :: mycomplex15
  end type extend15
  type, extends(extend15) :: extend16(my8int16)
    integer, len :: my8int16
    complex :: mycomplex16
  end type extend16
  type, extends(extend16) :: extend17(my8int17)
    integer, len :: my8int17
    complex :: mycomplex17
  end type extend17
  type, extends(extend17) :: extend18(my8int18)
    integer, len :: my8int18
    complex :: mycomplex18
  end type extend18
  type, extends(extend18) :: extend19(my8int19)
    integer, len :: my8int19
    complex :: mycomplex19
  end type extend19
  type, extends(extend19) :: extend20(my8int20)
    integer, len :: my8int20
    complex :: mycomplex20
  end type extend20
  type, extends(extend20) :: extend21(my8int21)
    integer, len :: my8int21
    complex :: mycomplex21
  end type extend21
  type, extends(extend21) :: extend22(my8int22)
    integer, len :: my8int22
    complex :: mycomplex22
  end type extend22
  type, extends(extend22) :: extend23(my8int23)
    integer, len :: my8int23
    complex :: mycomplex23
  end type extend23
  type, extends(extend23) :: extend24(my8int24)
    integer, len :: my8int24
    complex :: mycomplex24
  end type extend24
  type, extends(extend24) :: extend25(my8int25)
    integer, len :: my8int25
    complex :: mycomplex25
  end type extend25
  type, extends(extend25) :: extend26(my8int26)
    integer, len :: my8int26
    complex :: mycomplex26
  end type extend26
  type, extends(extend26) :: extend27(my8int27)
    integer, len :: my8int27
    complex :: mycomplex27
  end type extend27
  type, extends(extend27) :: extend28(my8int28)
    integer, len :: my8int28
    complex :: mycomplex28
  end type extend28
  type, extends(extend28) :: extend29(my8int29)
    integer, len :: my8int29
    complex :: mycomplex29
  end type extend29
  type, extends(extend29) :: extend30(my8int30)
    integer, len :: my8int30
    complex :: mycomplex30
  end type extend30
end module m

! Deep type extension chain where the base type is finalizable.
! Exercises IsFinalizable() returning true through the chain.
module m_final
  implicit none
  type :: fbase
    integer :: val
  contains
    final :: fbase_final
  end type fbase
  type, extends(fbase) :: fext2(fp2)
    integer, len :: fp2
    complex :: fc2
  end type fext2
  type, extends(fext2) :: fext3(fp3)
    integer, len :: fp3
    complex :: fc3
  end type fext3
  type, extends(fext3) :: fext4(fp4)
    integer, len :: fp4
    complex :: fc4
  end type fext4
  type, extends(fext4) :: fext5(fp5)
    integer, len :: fp5
    complex :: fc5
  end type fext5
  type, extends(fext5) :: fext6(fp6)
    integer, len :: fp6
    complex :: fc6
  end type fext6
  type, extends(fext6) :: fext7(fp7)
    integer, len :: fp7
    complex :: fc7
  end type fext7
  type, extends(fext7) :: fext8(fp8)
    integer, len :: fp8
    complex :: fc8
  end type fext8
  type, extends(fext8) :: fext9(fp9)
    integer, len :: fp9
    complex :: fc9
  end type fext9
  type, extends(fext9) :: fext10(fp10)
    integer, len :: fp10
    complex :: fc10
  end type fext10
  type, extends(fext10) :: fext11(fp11)
    integer, len :: fp11
    complex :: fc11
  end type fext11
  type, extends(fext11) :: fext12(fp12)
    integer, len :: fp12
    complex :: fc12
  end type fext12
  type, extends(fext12) :: fext13(fp13)
    integer, len :: fp13
    complex :: fc13
  end type fext13
  type, extends(fext13) :: fext14(fp14)
    integer, len :: fp14
    complex :: fc14
  end type fext14
  type, extends(fext14) :: fext15(fp15)
    integer, len :: fp15
    complex :: fc15
  end type fext15
  type, extends(fext15) :: fext16(fp16)
    integer, len :: fp16
    complex :: fc16
  end type fext16
  type, extends(fext16) :: fext17(fp17)
    integer, len :: fp17
    complex :: fc17
  end type fext17
  type, extends(fext17) :: fext18(fp18)
    integer, len :: fp18
    complex :: fc18
  end type fext18
  type, extends(fext18) :: fext19(fp19)
    integer, len :: fp19
    complex :: fc19
  end type fext19
  type, extends(fext19) :: fext20(fp20)
    integer, len :: fp20
    complex :: fc20
  end type fext20
  type, extends(fext20) :: fext21(fp21)
    integer, len :: fp21
    complex :: fc21
  end type fext21
  type, extends(fext21) :: fext22(fp22)
    integer, len :: fp22
    complex :: fc22
  end type fext22
  type, extends(fext22) :: fext23(fp23)
    integer, len :: fp23
    complex :: fc23
  end type fext23
  type, extends(fext23) :: fext24(fp24)
    integer, len :: fp24
    complex :: fc24
  end type fext24
  type, extends(fext24) :: fext25(fp25)
    integer, len :: fp25
    complex :: fc25
  end type fext25
  type, extends(fext25) :: fext26(fp26)
    integer, len :: fp26
    complex :: fc26
  end type fext26
  type, extends(fext26) :: fext27(fp27)
    integer, len :: fp27
    complex :: fc27
  end type fext27
  type, extends(fext27) :: fext28(fp28)
    integer, len :: fp28
    complex :: fc28
  end type fext28
  type, extends(fext28) :: fext29(fp29)
    integer, len :: fp29
    complex :: fc29
  end type fext29
  type, extends(fext29) :: fext30(fp30)
    integer, len :: fp30
    complex :: fc30
  end type fext30
contains
  subroutine fbase_final(x)
    type(fbase) :: x
  end subroutine
end module m_final
