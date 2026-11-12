!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
module m
  generic :: gen => spec2, spec1
 contains
  subroutine spec2(a,b)
    real, intent(in) :: a(*), b(*)
  end
  subroutine spec1(a)
    real, intent(in) :: a(:)
  end
  subroutine test(a)
    real, intent(in) :: a(*)
!CHECK: Dummy argument 'b=' (#2) is not OPTIONAL and is not associated with an actual argument in this procedure reference
!CHECK: Assumed-size array may not be associated with assumed-shape dummy argument 'a='
    call gen(a)
  end
end
