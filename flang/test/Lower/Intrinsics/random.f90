! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPrandom_test_1
subroutine random_test_1
  ! CHECK-DAG: hlfir.declare {{.*}}random_test_1Ess
  ! CHECK-DAG: hlfir.declare {{.*}}random_test_1Evv
  integer ss, vv(40)
  ! CHECK-DAG: hlfir.declare {{.*}}random_test_1Err
  ! CHECK-DAG: hlfir.declare {{.*}}random_test_1Eaa
  real rr, aa(5)
  ! CHECK: fir.call @_FortranARandomInit(%true{{.*}}, %false{{.*}}) {{.*}}: (i1, i1) -> ()
  call random_init(.true., .false.)
  ! CHECK: fir.call @_FortranARandomSeedSize(
  call random_seed(size=ss)
  print*, 'size: ', ss
  ! CHECK: fir.call @_FortranARandomSeedDefaultPut() {{.*}}: () -> ()
  call random_seed()
  ! CHECK: fir.call @_FortranARandomNumber(
  call random_number(rr)
  print*, rr
  ! CHECK: fir.call @_FortranARandomSeedGet(
  call random_seed(get=vv)
! print*, 'get:  ', vv(1:ss)
  ! CHECK: fir.call @_FortranARandomSeedPut(
  call random_seed(put=vv)
  print*, 'put:  ', vv(1:ss)
  ! CHECK: fir.call @_FortranARandomNumber(
  call random_number(aa)
  print*, aa
end

! CHECK-LABEL: func @_QPrandom_test_2
subroutine random_test_2
  integer :: size, get(5) = -9
  call foo(size)
  call bar(size, get)
contains
  ! CHECK-LABEL: func private @_QFrandom_test_2Pfoo
  subroutine foo(size, put, get)
    integer, optional :: size
    integer, optional :: put(5)
    integer, optional :: get(5)
    ! CHECK: [[s1:%[0-9]+]] = fir.is_present
    ! CHECK: [[p1:%[0-9]+]] = fir.is_present
    ! CHECK: [[g1:%[0-9]+]] = fir.is_present
    ! CHECK: [[s2:%[0-9]+]] = fir.embox
    ! CHECK: [[s3:%[0-9]+]] = fir.absent !fir.box<i32>
    ! CHECK: [[s4:%[0-9]+]] = arith.select [[s1]], [[s2]], [[s3]] : !fir.box<i32>
    ! CHECK: [[p2:%[0-9]+]] = fir.embox
    ! CHECK: [[p3:%[0-9]+]] = fir.absent !fir.box<!fir.array<5xi32>>
    ! CHECK: [[p4:%[0-9]+]] = arith.select [[p1]], [[p2]], [[p3]] : !fir.box<!fir.array<5xi32>>
    ! CHECK: [[g2:%[0-9]+]] = fir.embox
    ! CHECK: [[g3:%[0-9]+]] = fir.absent !fir.box<!fir.array<5xi32>>
    ! CHECK: [[g4:%[0-9]+]] = arith.select [[g1]], [[g2]], [[g3]] : !fir.box<!fir.array<5xi32>>
    ! CHECK: [[s5:%[0-9]+]] = fir.convert [[s4]] : (!fir.box<i32>) -> !fir.box<none>
    ! CHECK: [[p5:%[0-9]+]] = fir.convert [[p4]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
    ! CHECK: [[g5:%[0-9]+]] = fir.convert [[g4]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
    ! CHECK: fir.call @_FortranARandomSeed([[s5]], [[p5]], [[g5]]
    call random_seed(size, put, get)
    print*, size
  end subroutine

  ! CHECK-LABEL: func private @_QFrandom_test_2Pbar
  subroutine bar(size, get, put)
    integer, optional :: size
    integer, optional :: put(5)
    integer, optional :: get(5)
    ! CHECK: [[p1:%[0-9]+]] = fir.is_present
    ! CHECK: [[g1:%[0-9]+]] = fir.is_present
    ! CHECK: [[p2:%[0-9]+]] = fir.embox
    ! CHECK: [[p3:%[0-9]+]] = fir.absent !fir.box<!fir.array<5xi32>>
    ! CHECK: [[p4:%[0-9]+]] = arith.select [[p1]], [[p2]], [[p3]] : !fir.box<!fir.array<5xi32>>
    ! CHECK: [[g2:%[0-9]+]] = fir.embox
    ! CHECK: [[g3:%[0-9]+]] = fir.absent !fir.box<!fir.array<5xi32>>
    ! CHECK: [[g4:%[0-9]+]] = arith.select [[g1]], [[g2]], [[g3]] : !fir.box<!fir.array<5xi32>>
    ! CHECK: [[s1:%[0-9]+]] = fir.absent !fir.box<none>
    ! CHECK: [[p5:%[0-9]+]] = fir.convert [[p4]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
    ! CHECK: [[g5:%[0-9]+]] = fir.convert [[g4]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
    ! CHECK: fir.call @_FortranARandomSeed([[s1]], [[p5]], [[g5]]
    call random_seed(put=put, get=get)
    print*, get(1:size+1) ! "extra" value should be -9
  end subroutine
end

  call random_test_1
  call random_test_2
end
