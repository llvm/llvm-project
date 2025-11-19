! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

module mm
  implicit none
  type two
     integer(4) :: a, b
  end type two

  type three
     integer(8) :: a, b, c
  end type three

  type twothree
     type(two) t2
     type(three) t3
  end type twothree

contains
!CHECK-LABEL: Subprogram scope: inittwo
  subroutine inittwo(x,n)
    integer :: n
    type(two) :: x
    x%a=n
    x%b=n
  end subroutine inittwo
  
  subroutine initthree(x,n)
    integer :: n
    type(three) :: x
    x%a=n
    x%b=n
  end subroutine initthree

  function add_two(x, y)
    type(two) add_two, x, y, res
    res%a = x%a + y%a
    res%b = x%b + y%b
    add_two = res
  end function add_two

  function add_three(x, y)
    type(three) add_three, x, y, res
    res%a = x%a + y%a
    res%b = x%b + y%b
    res%c = x%c + y%c
    add_three = res
  end function add_three
  
!CHECK-LABEL: Subprogram scope: functwo
  function functwo(x, n)
    type(two) functwo
    integer :: n
    type(two) ::  x(n)
    type(two) :: res
    integer :: i
    !$omp declare reduction(adder:two:omp_out=add_two(omp_out,omp_in)) initializer(inittwo(omp_priv,0))
!CHECK: adder: UserReductionDetails TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_in size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_out size=8 offset=8: ObjectEntity type: TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_orig size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_priv size=8 offset=8: ObjectEntity type: TYPE(two)
    
  
    !$omp simd reduction(adder:res)
    do i=1,n
       res=add_two(res,x(i))
    enddo
    functwo=res
  end function functwo

  function functhree(x, n)
    implicit none
    type(three) :: functhree
    type(three) :: x(n)
    type(three) :: res
    integer :: i
    integer :: n
    !$omp declare reduction(adder:three:omp_out=add_three(omp_out,omp_in)) initializer(initthree(omp_priv,1))
    
    !$omp simd reduction(adder:res)
    do i=1,n
       res=add_three(res,x(i))
    enddo
    functhree=res
  end function functhree
  
  function functwothree(x, n)
    type(twothree) :: functwothree
    type(twothree) :: x(n)
    type(twothree) :: res
    type(two) :: res2
    type(three) :: res3
    integer :: n
    integer :: i

    !$omp declare reduction(adder:two:omp_out=add_two(omp_out,omp_in)) initializer(inittwo(omp_priv,0))
    
    !$omp declare reduction(adder:three:omp_out=add_three(omp_out,omp_in)) initializer(initthree(omp_priv,1))
    
!CHECK: adder: UserReductionDetails TYPE(two) TYPE(three)
!CHECK OtherConstruct scope
!CHECK: omp_in size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_out size=8 offset=8: ObjectEntity type: TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_orig size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_priv size=8 offset=8: ObjectEntity type: TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_in size=24 offset=0: ObjectEntity type: TYPE(three)
!CHECK: omp_out size=24 offset=24: ObjectEntity type: TYPE(three)
!CHECK OtherConstruct scope
!CHECK: omp_orig size=24 offset=0: ObjectEntity type: TYPE(three)
!CHECK: omp_priv size=24 offset=24: ObjectEntity type: TYPE(three)

    !$omp simd reduction(adder:res3)
    do i=1,n
       res3=add_three(res%t3,x(i)%t3)
    enddo

    !$omp simd reduction(adder:res2)
    do i=1,n
       res2=add_two(res2,x(i)%t2)
    enddo
    res%t2 = res2
    res%t3 = res3
    functwothree=res
  end function functwothree

!CHECK-LABEL: Subprogram scope: funcbtwo
  function funcBtwo(x, n)
    type(two) funcBtwo
    integer :: n
    type(two) ::  x(n)
    type(two) :: res
    integer :: i
    !$omp declare reduction(+:two:omp_out=add_two(omp_out,omp_in)) initializer(inittwo(omp_priv,0))
!CHECK: op.+: UserReductionDetails TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_in size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_out size=8 offset=8: ObjectEntity type: TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_orig size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_priv size=8 offset=8: ObjectEntity type: TYPE(two)
    
  
    !$omp simd reduction(+:res)
    do i=1,n
       res=add_two(res,x(i))
    enddo
    funcBtwo=res
  end function funcBtwo

  function funcBtwothree(x, n)
    type(twothree) :: funcBtwothree
    type(twothree) :: x(n)
    type(twothree) :: res
    type(two) :: res2
    type(three) :: res3
    integer :: n
    integer :: i

    !$omp declare reduction(+:two:omp_out=add_two(omp_out,omp_in)) initializer(inittwo(omp_priv,0))
    
    !$omp declare reduction(+:three:omp_out=add_three(omp_out,omp_in)) initializer(initthree(omp_priv,1))
    
!CHECK: op.+: UserReductionDetails TYPE(two) TYPE(three)
!CHECK OtherConstruct scope
!CHECK: omp_in size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_out size=8 offset=8: ObjectEntity type: TYPE(two)
!CHECK OtherConstruct scope
!CHECK: omp_orig size=8 offset=0: ObjectEntity type: TYPE(two)
!CHECK: omp_priv size=8 offset=8: ObjectEntity type: TYPE(two)
!CHECK: OtherConstruct scope
!CHECK: omp_in size=24 offset=0: ObjectEntity type: TYPE(three)
!CHECK: omp_out size=24 offset=24: ObjectEntity type: TYPE(three)
!CHECK OtherConstruct scope
!CHECK: omp_orig size=24 offset=0: ObjectEntity type: TYPE(three)
!CHECK: omp_priv size=24 offset=24: ObjectEntity type: TYPE(three)

    !$omp simd reduction(+:res3)
    do i=1,n
       res3=add_three(res%t3,x(i)%t3)
    enddo

    !$omp simd reduction(+:res2)
    do i=1,n
       res2=add_two(res2,x(i)%t2)
    enddo
    res%t2 = res2
    res%t3 = res3
    funcBtwothree = res
  end function funcBtwothree

  !! This is checking a special case, where a reduction is declared inside a
  !! pure function

  pure logical function reduction()
!CHECK: reduction size=4 offset=0: ObjectEntity funcResult type: LOGICAL(4)
!CHECK: rr: UserReductionDetails INTEGER(4)
!CHECK: OtherConstruct scope: size=8 alignment=4 sourceRange=0 bytes
!CHECK: omp_in size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK: omp_out size=4 offset=4: ObjectEntity type: INTEGER(4)
!CHECK: OtherConstruct scope: size=8 alignment=4 sourceRange=0 bytes
!CHECK: omp_orig size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK: omp_priv size=4 offset=4: ObjectEntity type: INTEGER(4)
    !$omp declare reduction (rr : integer : omp_out = omp_out + omp_in) initializer (omp_priv = 0)
    reduction = .false.
  end function reduction
  
end module mm
