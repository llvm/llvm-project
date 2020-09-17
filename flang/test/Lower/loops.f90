! RUN: bbc -emit-fir -o - %s | FileCheck %s

  ! CHECK-DAG: fir.alloca !fir.array<5x5x5xi32> {name = "a"}
  ! CHECK-DAG: fir.alloca i8 {name = "i"}
  ! CHECK-DAG: fir.alloca i16 {name = "i"}
  ! CHECK-DAG: fir.alloca i32 {name = "i"}
  ! CHECK-DAG: fir.alloca i32 {name = "i"}
  ! CHECK-DAG: fir.alloca i8 {name = "j"}
  ! CHECK-DAG: fir.alloca i32 {name = "j"}
  ! CHECK-DAG: fir.alloca i32 {name = "j"}
  ! CHECK-DAG: fir.alloca i8 {name = "k"}
  ! CHECK-DAG: fir.alloca i32 {name = "k"}
  ! CHECK-DAG: fir.alloca i32 {name = "k"}
  integer(4) :: a(5,5,5), i, j, k, asum, xsum

  i = 100
  j = 200
  k = 300

  ! CHECK-COUNT-3: fir.do_loop {{.*}} unordered
  do concurrent (i=1:5, j=1:5, k=1:5) ! shared(a)
    ! CHECK: fir.coordinate_of
    a(i,j,k) = 0
  enddo
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  print*, i, j, k

  ! CHECK-COUNT-3: fir.do_loop {{.*}} unordered
  do concurrent (integer(1)::i=1:5, j=1:5, k=1:5, i.ne.j .and. k.ne.3) shared(a)
    ! CHECK-COUNT-2: fir.coordinate_of
    a(i,j,k) = a(i,j,k) + 1
  enddo

  ! CHECK-COUNT-3: fir.do_loop {{[^un]*}} -> (index)
  asum = 0
  do i=1,5
    do j=1,5
      do k=1,5
        ! CHECK: fir.coordinate_of
        asum = asum + a(i,j,k)
      enddo
    enddo
  enddo
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  print*, i, j, k, '-', asum

  ! CHECK-NOT: fir.do_loop
  do concurrent (integer(2)::i=1:5, i.ne.3)
    if (i.eq.2 .or. i.eq.4) goto 9
    ! CHECK: fir.call @_FortranAioBeginExternalListOutput
    print*, i
  9 continue
  enddo
  
  xsum = 0.0
  ! CHECK-NOT: fir.do_loop
  do x = 1.5, 3.5, 0.3
    xsum = xsum + 1
  enddo
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedOutput
  print '(X,F3.1,A,I2)', x, ' -', xsum
end
