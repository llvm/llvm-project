! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! The special variables omp_in, omp_out, omp_priv and omp_orig in DECLARE
! REDUCTION are scalars: any selectors in their designators must be component
! selectors.

module m
  type t
    integer :: x
    integer :: a(4)
  end type
contains
  subroutine comb(o, i)
    type(t) :: o, i
    o%x = o%x + i%x
  end
  subroutine init(p, q)
    type(t) :: p, q
    p = q
  end
  pure integer function f(a, b)
    integer, intent(in) :: a, b
    f = a + b
  end
end module

subroutine valid
  use m
  !$omp declare reduction(r1:t:omp_out%x = omp_out%x + omp_in%x) &
  !$omp& initializer(omp_priv = t(0, 0))
  !$omp declare reduction(r2:t:comb(omp_out, omp_in)) &
  !$omp& initializer(init(omp_priv, omp_orig))
  !$omp declare reduction(r3:integer:omp_out = f(omp_out, omp_in)) &
  !$omp& initializer(omp_priv = omp_orig)
  !$omp declare reduction(r4:t:omp_out%a(1) = omp_out%a(1) + omp_in%a(2)) &
  !$omp& initializer(omp_priv = omp_orig)
end

subroutine invalid
  interface
    subroutine s1(k)
      integer :: k
    end
    subroutine s2(k, n)
      integer :: k, n
    end
  end interface
  !ERROR: 'omp_out' is not an array
  !$omp declare reduction(aaa:integer:s1(omp_out(1))) initializer(omp_priv=0)
  !ERROR: 'omp_in' is not an array
  !$omp declare reduction(bbb:integer:s2(omp_out,omp_in(1))) initializer(omp_priv=0)
  !ERROR: 'omp_in' is not an array
  !$omp declare reduction(ccc:integer:omp_out=omp_in(1)) initializer(omp_priv=0)
  !ERROR: 'omp_out' is not an array
  !$omp declare reduction(ddd:integer:omp_out(1)=omp_in) initializer(omp_priv=0)
  !ERROR: 'omp_out' is not an array
  !$omp declare reduction(eee:integer:omp_out=omp_in+omp_out(:)) initializer(omp_priv=0)
  !ERROR: 'omp_orig' is not an array
  !$omp declare reduction(fff:integer:omp_out=omp_in) initializer(omp_priv=omp_orig(1))
  !ERROR: 'omp_priv' is not an array
  !$omp declare reduction(ggg:integer:omp_out=omp_in) initializer(s1(omp_priv(1)))
end
