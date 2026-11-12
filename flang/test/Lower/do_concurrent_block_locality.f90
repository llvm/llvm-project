! Test variables declared in a BLOCK that appear in a locality-spec of a
! DO CONCURRENT inside that BLOCK: the locality operands are the BLOCK's
! variables.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

subroutine block_locality(n, r)
  integer :: n, r, j
  block
    integer :: t1, t2, t3
    t2 = 1
    t3 = 0
    do concurrent (j=1:n) local(t1) local_init(t2) reduce(+:t3)
      t1 = j + t2
      t3 = t3 + t1
    end do
    r = t3
  end block
end subroutine

! CHECK-DAG: fir.local {type = local_init} @[[T2_PRIV:_QFblock_localityB1Et2_firstprivate_i32]] : i32
! CHECK-DAG: fir.local {type = local} @[[T1_PRIV:_QFblock_localityB1Et1_private_i32]] : i32

! CHECK-LABEL: func.func @_QPblock_locality(
! CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFblock_localityB1Et1")
! CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFblock_localityB1Et2")
! CHECK:         %[[T3:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFblock_localityB1Et3")
! CHECK:         fir.do_concurrent.loop
! CHECK-SAME:      local(@[[T1_PRIV]] %[[T1]]#0 -> %{{[a-z0-9]+}}, @[[T2_PRIV]] %[[T2]]#0 -> %{{[a-z0-9]+}} :
! CHECK-SAME:      reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[T3]]#0 -> %{{[a-z0-9]+}} :

subroutine nested_block_locality(n)
  integer :: n, j
  block
    integer :: t5
    block
      do concurrent (j=1:n) local(t5)
        t5 = j
      end do
    end block
  end block
end subroutine

! CHECK-LABEL: func.func @_QPnested_block_locality(
! CHECK:         %[[T5:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFnested_block_localityB1Et5")
! CHECK:         fir.do_concurrent.loop
! CHECK-SAME:      local(@_QFnested_block_localityB1Et5_private_i32 %[[T5]]#0 -> %{{[a-z0-9]+}} :
