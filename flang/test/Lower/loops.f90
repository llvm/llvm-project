! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: loop_test
subroutine loop_test
  ! CHECK: %[[VAL_2:.*]] = fir.alloca i16 {bindc_name = "i"}
  ! CHECK: %[[VAL_3:.*]] = fir.alloca i16 {bindc_name = "i"}
  ! CHECK: %[[VAL_4:.*]] = fir.alloca i16 {bindc_name = "i"}
  ! CHECK: %[[VAL_5:.*]] = fir.alloca i8 {bindc_name = "k"}
  ! CHECK: %[[VAL_6:.*]] = fir.alloca i8 {bindc_name = "j"}
  ! CHECK: %[[VAL_7:.*]] = fir.alloca i8 {bindc_name = "i"}
  ! CHECK: %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "k"}
  ! CHECK: %[[VAL_9:.*]] = fir.alloca i32 {bindc_name = "j"}
  ! CHECK: %[[VAL_10:.*]] = fir.alloca i32 {bindc_name = "i"}
  ! CHECK: %[[VAL_11:.*]] = fir.alloca !fir.array<5x5x5xi32> {bindc_name = "a", uniq_name = "_QFloop_testEa"}
  ! CHECK: %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "asum", uniq_name = "_QFloop_testEasum"}
  ! CHECK: %[[VAL_13:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFloop_testEi"}
  ! CHECK: %[[VAL_14:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFloop_testEj"}
  ! CHECK: %[[VAL_15:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFloop_testEk"}
  ! CHECK: %[[VAL_16:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFloop_testEx"}
  ! CHECK: %[[VAL_17:.*]] = fir.alloca i32 {bindc_name = "xsum", uniq_name = "_QFloop_testExsum"}

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
  print*, 'A:', i, j, k

  ! CHECK-COUNT-3: fir.do_loop {{.*}} unordered
  ! CHECK: fir.if
  do concurrent (integer(1)::i=1:5, j=1:5, k=1:5, i.ne.j .and. k.ne.3) shared(a)
    ! CHECK-COUNT-2: fir.coordinate_of
    a(i,j,k) = a(i,j,k) + 1
  enddo

  ! CHECK-COUNT-3: fir.do_loop {{[^un]*}} -> (index, i32)
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
  print*, 'B:', i, j, k, '-', asum

  ! CHECK: fir.do_loop {{.*}} unordered
  ! CHECK-COUNT-2: fir.if
  do concurrent (integer(2)::i=1:5, i.ne.3)
    if (i.eq.2 .or. i.eq.4) goto 5 ! fir.if
    ! CHECK: fir.call @_FortranAioBeginExternalListOutput
    print*, 'C:', i
  5 continue
  enddo

  ! CHECK: fir.do_loop {{.*}} unordered
  ! CHECK-COUNT-2: fir.if
  do concurrent (integer(2)::i=1:5, i.ne.3)
    if (i.eq.2 .or. i.eq.4) then ! fir.if
      goto 6
    endif
    ! CHECK: fir.call @_FortranAioBeginExternalListOutput
    print*, 'D:', i
  6 continue
  enddo

  ! CHECK-NOT: fir.do_loop
  ! CHECK-NOT: fir.if
  do concurrent (integer(2)::i=1:5, i.ne.3)
    goto (7, 7) i+1
    ! CHECK: fir.call @_FortranAioBeginExternalListOutput
    print*, 'E:', i
  7 continue
  enddo

  xsum = 0.0
  ! CHECK-NOT: fir.do_loop
  do x = 1.5, 3.5, 0.3
    xsum = xsum + 1
  enddo
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedOutput
  print '(" F:",X,F3.1,A,I2)', x, ' -', xsum
end subroutine loop_test

! CHECK-LABEL: c.func @_QPlis
subroutine lis(n)
  ! CHECK-DAG: fir.alloca i32 {bindc_name = "m"}
  ! CHECK-DAG: fir.alloca i32 {bindc_name = "j"}
  ! CHECK-DAG: fir.alloca i32 {bindc_name = "i"}
  ! CHECK-DAG: fir.alloca i8 {bindc_name = "i"}
  ! CHECK-DAG: fir.alloca i32 {bindc_name = "j", uniq_name = "_QFlisEj"}
  ! CHECK-DAG: fir.alloca i32 {bindc_name = "k", uniq_name = "_QFlisEk"}
  ! CHECK-DAG: fir.alloca !fir.box<!fir.ptr<!fir.array<?x?x?xi32>>> {bindc_name = "p", uniq_name = "_QFlisEp"}
  ! CHECK-DAG: fir.alloca !fir.array<?x?x?xi32>, %{{.*}}, %{{.*}}, %{{.*}} {bindc_name = "a", fir.target, uniq_name = "_QFlisEa"}
  ! CHECK-DAG: fir.alloca !fir.array<?x?xi32>, %{{.*}}, %{{.*}} {bindc_name = "r", uniq_name = "_QFlisEr"}
  ! CHECK-DAG: fir.alloca !fir.array<?x?xi32>, %{{.*}}, %{{.*}} {bindc_name = "s", uniq_name = "_QFlisEs"}
  ! CHECK-DAG: fir.alloca !fir.array<?x?xi32>, %{{.*}}, %{{.*}} {bindc_name = "t", uniq_name = "_QFlisEt"}
  integer, target    :: a(n,n,n) ! operand via p
  integer            :: r(n,n)   ! result, unspecified locality
  integer            :: s(n,n)   ! shared locality
  integer            :: t(n,n)   ! local locality
  integer, pointer   :: p(:,:,:) ! local_init locality

  p => a
  ! CHECK:     fir.do_loop %arg1 = %c0{{.*}} to %{{.*}} step %c1{{.*}} unordered iter_args(%arg2 = %{{.*}}) -> (!fir.array<?x?xi32>) {
  ! CHECK:       fir.do_loop %arg3 = %c0{{.*}} to %{{.*}} step %c1{{.*}} unordered iter_args(%arg4 = %arg2) -> (!fir.array<?x?xi32>) {
  ! CHECK:       }
  ! CHECK:     }
  r = 0

  ! CHECK:     fir.do_loop %arg1 = %{{.*}} to %{{.*}} step %{{.*}} unordered {
  ! CHECK:       fir.do_loop %arg2 = %{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%arg3 = %{{.*}}) -> (index, i32) {
  ! CHECK:       }
  ! CHECK:     }
  do concurrent (integer(kind=1)::i=n:1:-1)
    do j = 1,n
      a(i,j,:) = 2*(i+j)
      s(i,j)   = -i-j
    enddo
  enddo

  ! CHECK:     fir.do_loop %arg1 = %{{.*}} to %{{.*}} step %c1{{.*}} unordered {
  ! CHECK:       fir.do_loop %arg2 = %{{.*}} to %{{.*}} step %c1{{.*}} unordered {
  ! CHECK:         fir.if %{{.*}} {
  ! CHECK:           %[[V_95:[0-9]+]] = fir.alloca !fir.array<?x?xi32>, %{{.*}}, %{{.*}} {bindc_name = "t", pinned, uniq_name = "_QFlisEt"}
  ! CHECK:           %[[V_96:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?x?xi32>>> {bindc_name = "p", pinned, uniq_name = "_QFlisEp"}
  ! CHECK:           fir.store %{{.*}} to %[[V_96]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>>
  ! CHECK:           fir.do_loop %arg3 = %{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%arg4 = %{{.*}}) -> (index, i32) {
  ! CHECK:             fir.do_loop %arg5 = %{{.*}} to %{{.*}} step %c1{{.*}} unordered {
  ! CHECK:               fir.load %[[V_96]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>>
  ! CHECK:               fir.convert %[[V_95]] : (!fir.ref<!fir.array<?x?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  ! CHECK:             }
  ! CHECK:           }
  ! CHECK:           fir.convert %[[V_95]] : (!fir.ref<!fir.array<?x?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  ! CHECK:         }
  ! CHECK:       }
  ! CHECK:     }
  do concurrent (i=1:n,j=1:n,i.ne.j) local(t) local_init(p) shared(s)
    do k=1,n
      do concurrent (m=1:n)
        t(k,m) = p(k,m,k)
      enddo
    enddo
    r(i,j) = t(i,j) + s(i,j)
  enddo

  print*, sum(r) ! n=6 -> 210
end

! CHECK-LABEL: print_nothing
subroutine print_nothing(k1, k2)
  if (k1 > 0) then
    ! CHECK: br [[header:\^bb[0-9]+]]
    ! CHECK: [[header]]
    do while (k1 > k2)
      print*, k1, k2 ! no output
      k2 = k2 + 1
      ! CHECK: br [[header]]
    end do
  end if
end

  call loop_test
  call lis(6)
  call print_nothing(2, 2)
end
