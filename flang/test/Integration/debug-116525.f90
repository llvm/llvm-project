! RUN: %flang_fc1 -fopenmp -emit-llvm -debug-info-kind=standalone %s -o -

! Test that this does not cause build failure.
function s(x)
    character(len=2) :: x, s, ss

    s = x

    entry ss()

end function s

