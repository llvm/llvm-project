! RUN: %python %S/test_errors.py %s %flang_fc1

module m1_r1223_6
contains
        function m1f1()
                integer :: m1f1
                real :: m1f1e1
                m1f1 = 0
                !ERROR: RESULT name 'm1f1e1' must be different from ENTRY name 'm1f1e1'
                entry m1f1e1() result(m1f1e1)
                m1f1e1 = 0.1
        end function
end module
program r1223c6
        use m1_r1223_6
end program
