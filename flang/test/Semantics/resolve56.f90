! RUN: %S/test_errors.sh %s %t %f18
! Test that associations constructs can be correctly combined. The intrinsic
! functions are not what is tested here, they are only use to reveal the types
! of local variables.

  implicit none
  real res
  complex zres
  integer ires
  class(*), allocatable :: a, b
  select type(a)
    type is (integer)
      select type(b)
        type is (integer)
          ires = selected_int_kind(b)
          ires = selected_int_kind(a)
      end select
    type is (real)
     res = acos(a)
     !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
     res = acos(b)
  end select

  select type(c => a)
    type is (real)
     res = acos(c)
    class default
     !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
     res = acos(c)
  end select
  select type(a)
    type is (integer)
     !ERROR: Actual argument for 'x=' has bad type 'INTEGER(4)'
     res = acos(a)
  end select

  select type(b)
    type is (integer)
      associate(y=>1.0, x=>1, z=>(1.0,2.3))
        ires = selected_int_kind(x)
        select type(a)
          type is (real)
            res = acos(a)
            res = acos(y)
            !ERROR: Actual argument for 'x=' has bad type 'INTEGER(4)'
            res = acos(b)
          type is (integer)
            ires = selected_int_kind(b)
            zres = acos(z)
           !ERROR: Actual argument for 'x=' has bad type 'INTEGER(4)'
           res = acos(a)
        end select
      end associate
      ires = selected_int_kind(b)
      !ERROR: No explicit type declared for 'c'
      ires = selected_int_kind(c)
      !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
      res = acos(a)
    class default
      !ERROR: Actual argument for 'r=' has bad type 'CLASS(*)'
      ires = selected_int_kind(b)
  end select
  !ERROR: Actual argument for 'r=' has bad type 'CLASS(*)'
  ires = selected_int_kind(a)
  !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
  res = acos(b)
end
