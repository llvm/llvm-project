! RUN: %python %S/test_errors.py %s %flang_fc1
program io_advance_rec
    implicit none
    integer :: arr(10)

    open(10, file="dummy.dat", access='direct', recl=80)
    !ERROR: If ADVANCE appears, REC must not appear
    read(10, '(I1)', advance="no", rec=1, err=100) arr(1)

100 continue

    close(10)
end program