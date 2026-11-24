! Test references to a variable that is LOCAL or LOCAL_INIT in an enclosing
! DO CONCURRENT from a nested DO CONCURRENT locality-spec or from a BLOCK. They
! must use the enclosing construct's copy of the variable, not the variable of
! the subprogram.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -mmlir --enable-delayed-privatization=false \
! RUN:   -o - %s | FileCheck %s --check-prefix=NODELAY

subroutine reduce_in_local(n, nz, work)
  implicit none
  integer :: n, nz, work(nz), j, k, tmp
  do concurrent (k=1:nz) local(tmp)
    tmp = 0
    do concurrent (j=1:n) reduce(max:tmp)
      tmp = max(tmp, j)
    end do
    work(k) = tmp
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduce_in_local(
! CHECK:         %[[TMP:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFreduce_in_localEtmp")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %[[TMP]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFreduce_in_localEtmp")
! CHECK:           fir.do_concurrent.loop {{.*}} reduce(@max_reduction_i32 #fir.reduce_attr<max> %[[LOCAL]]#0 -> %[[RED_ARG:.*]] : !fir.ref<i32>) {
! CHECK:             %[[RED:.*]]:2 = hlfir.declare %[[RED_ARG]] uniq_name("_QFreduce_in_localEtmp")
! CHECK:             hlfir.assign %{{.*}} to %[[RED]]#0
! CHECK:           }
! CHECK:           fir.load %[[LOCAL]]#0

! NODELAY-LABEL: func.func @_QPreduce_in_local(
! NODELAY:         fir.do_concurrent.loop
! NODELAY:           %[[CLONE:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFreduce_in_localEtmp"
! NODELAY:           %[[LOCAL:.*]]:2 = hlfir.declare %[[CLONE]] uniq_name("_QFreduce_in_localEtmp")
! NODELAY:           fir.do_concurrent.loop {{.*}} reduce(@max_reduction_i32 #fir.reduce_attr<max> %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

subroutine reduce_in_local_init(n, nz, work)
  implicit none
  integer :: n, nz, work(nz), j, k, tmp
  tmp = 0
  do concurrent (k=1:nz) local_init(tmp)
    do concurrent (j=1:n) reduce(+:tmp)
      tmp = tmp + j
    end do
    work(k) = tmp
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduce_in_local_init(
! CHECK:         %[[TMP:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFreduce_in_local_initEtmp")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %[[TMP]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFreduce_in_local_initEtmp")
! CHECK:           fir.do_concurrent.loop {{.*}} reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

! NODELAY-LABEL: func.func @_QPreduce_in_local_init(
! NODELAY:         fir.do_concurrent.loop
! NODELAY:           %[[CLONE:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFreduce_in_local_initEtmp"
! NODELAY:           %[[LOCAL:.*]]:2 = hlfir.declare %[[CLONE]] uniq_name("_QFreduce_in_local_initEtmp")
! NODELAY:           fir.do_concurrent.loop {{.*}} reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

! Two levels of REDUCE under a LOCAL.
subroutine reduce_two_levels(n, work)
  implicit none
  integer :: n, i, j, k, t, work(n)
  do concurrent (k=1:n) local(t)
    t = 10*k
    do concurrent (i=1:2) reduce(+:t)
      do concurrent (j=1:4) reduce(+:t)
        t = t + j
      end do
    end do
    work(k) = t
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduce_two_levels(
! CHECK:         %[[T:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFreduce_two_levelsEt")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %[[T]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFreduce_two_levelsEt")
! CHECK:           fir.do_concurrent.loop {{.*}} reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[LOCAL]]#0 -> %[[RED1_ARG:.*]] : !fir.ref<i32>) {
! CHECK:             %[[RED1:.*]]:2 = hlfir.declare %[[RED1_ARG]] uniq_name("_QFreduce_two_levelsEt")
! CHECK:             fir.do_concurrent.loop {{.*}} reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[RED1]]#0 -> %{{.*}} : !fir.ref<i32>) {

! A LOCAL_INIT copy is initialized from the enclosing construct's copy
! (F2023 11.1.7.5), with and without delayed privatization.
subroutine local_init_in_local(n, inner)
  implicit none
  integer :: n, j, k, w, inner(2, n)
  w = 1
  do concurrent (k=1:n) local(w)
    w = 100 + k
    do concurrent (j=1:2) local_init(w)
      inner(j, k) = w
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlocal_init_in_local(
! CHECK:         %[[W:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFlocal_init_in_localEw")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@_QFlocal_init_in_localEw_private_i32 %[[W]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFlocal_init_in_localEw")
! CHECK:           fir.do_concurrent.loop {{.*}} local(@_QFlocal_init_in_localEw_firstprivate_i32 %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

! NODELAY-LABEL: func.func @_QPlocal_init_in_local(
! NODELAY:         fir.do_concurrent.loop
! NODELAY:           %[[OUTER:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFlocal_init_in_localEw"
! NODELAY:           %[[OUTER_DECL:.*]]:2 = hlfir.declare %[[OUTER]] uniq_name("_QFlocal_init_in_localEw")
! NODELAY:           fir.do_concurrent.loop
! NODELAY:             %[[INNER:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFlocal_init_in_localEw"
! NODELAY:             %[[INNER_DECL:.*]]:2 = hlfir.declare %[[INNER]] uniq_name("_QFlocal_init_in_localEw")
! NODELAY:             %[[INIT:.*]] = fir.load %[[OUTER_DECL]]#0
! NODELAY:             hlfir.assign %[[INIT]] to %[[INNER_DECL]]#0

! A LOCAL entity nested in a LOCAL_INIT entity is a LOCAL, not a LOCAL_INIT.
subroutine local_in_local_init(n, x)
  implicit none
  integer :: n, j, k, b, x(n, n)
  b = 7
  do concurrent (k=1:n) local_init(b)
    do concurrent (j=1:n) local(b)
      b = j
      x(j, k) = b
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlocal_in_local_init(
! CHECK:         %[[B:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFlocal_in_local_initEb")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@_QFlocal_in_local_initEb_firstprivate_i32 %[[B]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFlocal_in_local_initEb")
! CHECK:           fir.do_concurrent.loop {{.*}} local(@_QFlocal_in_local_initEb_private_i32 %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

! A BLOCK with a VOLATILE statement for a LOCAL variable refers to the
! enclosing construct's copy; it does not get storage of its own.
subroutine block_volatile_in_local(n, work)
  implicit none
  integer :: n, k, t, work(n)
  do concurrent (k=1:n) local(t)
    t = 10*k
    block
      volatile :: t
      t = t + 1
    end block
    work(k) = t
  end do
end subroutine

! CHECK-LABEL: func.func @_QPblock_volatile_in_local(
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %{{.*}} -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFblock_volatile_in_localEt")
! CHECK:           llvm.intr.stacksave
! CHECK-NOT:       fir.alloca {{.*}}uniq_name = "_QFblock_volatile_in_localEt"
! CHECK:           %[[OLD:.*]] = fir.load %[[LOCAL]]#0
! CHECK:           %[[NEW:.*]] = arith.addi %[[OLD]], %{{.*}}
! CHECK:           hlfir.assign %[[NEW]] to %[[LOCAL]]#0
! CHECK:           llvm.intr.stackrestore

! Likewise when the BLOCK makes the variable implicitly ASYNCHRONOUS.
subroutine block_asynchronous_in_local(n, u, work)
  implicit none
  integer :: n, u, k, t, work(n)
  do concurrent (k=1:n) local(t)
    t = 10*k
    block
      write(u, asynchronous='yes') t
      wait(u)
      t = t + 1
    end block
    work(k) = t
  end do
end subroutine

! CHECK-LABEL: func.func @_QPblock_asynchronous_in_local(
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %{{.*}} -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFblock_asynchronous_in_localEt")
! CHECK-NOT:       fir.alloca {{.*}}uniq_name = "_QFblock_asynchronous_in_localEt"
! CHECK:           fir.embox %[[LOCAL]]#0
! CHECK:           fir.call @_FortranAioOutputDescriptor
! CHECK:           hlfir.assign %{{.*}} to %[[LOCAL]]#0
