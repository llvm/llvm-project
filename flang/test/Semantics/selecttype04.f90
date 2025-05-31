! RUN: %python %S/test_errors.py %s %flang_fc1
! Check F'2023 C1167
module m
  type :: base(kindparam, lenparam)
    integer, kind :: kindparam
    integer, len :: lenparam
  end type
  type, extends(base) :: ext1
   contains
    procedure :: tbp
  end type
  type, extends(ext1) :: ext2
  end type
 contains
  function tbp(x)
    class(ext1(123,*)), target :: x
    class(ext1(123,:)), pointer :: tbp
    tbp => x
  end
  subroutine test
    type(ext1(123,456)), target :: var
    select type (sel => var%tbp())
    type is (ext1(123,*)) ! ok
    type is (ext2(123,*)) ! ok
    !ERROR: Type specification 'ext1(kindparam=234_4,lenparam=*)' must be an extension of TYPE 'ext1(kindparam=123_4,lenparam=:)'
    type is (ext1(234,*))
    !ERROR: Type specification 'ext2(kindparam=234_4,lenparam=*)' must be an extension of TYPE 'ext1(kindparam=123_4,lenparam=:)'
    type is (ext2(234,*))
    end select
  end
end
