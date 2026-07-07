! Check that appropriate errors are given when the SIMD Directive is used
! with no DO loop following

! RUN: %python %S/test_errors.py %s %flang -Werror

subroutine test()
!WARNING: A DO loop must follow the SIMD directive
  !DIR$ SIMD
end subroutine
