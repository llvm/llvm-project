! This test writes a large 4D array to an unformatted file using nested
! implied DO constructs. The updated lowering aims to produce fewer
! reallocations as the size increases geometrically when certain
! threshold is reached

! RUN: %flang %isysroot -L"%libdir" -O3 -g %s -o %t && %t | FileCheck %s
! CHECK: PASS

program test_array_write
    integer::u,i,j,k
    real (kind = 8), allocatable, dimension (:,:,:,:):: buf
    allocate(buf(128, 128, 128, 5))
    buf = 0
    open(newunit=u, file = "test_array_write_output.txt", &
         status="replace", form = "unformatted")
    write(u) &
        ((((buf(i,j,k,1)),i=1,127),j=1,127),k=1,127), &
        ((((buf(i,j,k,2)),i=1,127),j=1,127),k=1,127), &
        ((((buf(i,j,k,3)),i=1,127),j=1,127),k=1,127), &
        ((((buf(i,j,k,4)),i=1,127),j=1,127),k=1,127), &
        ((((buf(i,j,k,5)),i=1,127),j=1,127),k=1,127)
    print *, "PASS"
    close(u)
    deallocate(buf)
end program test_array_write
