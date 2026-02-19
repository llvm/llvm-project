! RUN: bbc -sentinel-test=sen$ --pft-test %s -o - 2>&1 | FileCheck %s

! Test support additional compiler directive sentinels.

module test_sentinel
  interface
  subroutine alternative_sentinel(a)
    integer(4) :: a
    !sen$ ignore_tkr a
    !CHECK: CompilerDirective: !ignore_tkr a
  end subroutine
end interface

end module test_sentinel
