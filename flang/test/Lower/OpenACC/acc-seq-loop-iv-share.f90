! The induction variable of an unstructured (goto/exit) `acc.loop seq` is not
! privatized on the loop itself, so the value it has after the loop is seen by
! the code that follows. The enclosing loop/compute construct privatizes it.
! -openacc-share-seq-loop-iv=false restores the old behavior.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=ALL,SHARED
! RUN: bbc -fopenacc -emit-hlfir -openacc-share-seq-loop-iv=false %s -o - | FileCheck %s --check-prefixes=ALL,PRIVATE

! The loop exited by goto reads NuGE afterwards; later vector loops reuse the
! same variable and still get their own private copy.
subroutine gaussian_shape(a, mu, nu, r)
  integer :: mu, nu, mu0, nuge, nu1
  real :: a(3, 3), r(2, 3)
  !$acc parallel
  !$acc loop gang
  do mu0 = 1, 2
    !$acc loop seq
    do 40 nuge = 1, 3
      if (nu .ge. mu) goto 50
40  continue
50  nu1 = nuge
    !$acc loop vector
    do nuge = 1, nu1 - 1
      r(mu0, nuge) = a(mu0, nuge)
    end do
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPgaussian_shape
! ALL:         acc.loop gang private({{.*}}) control
! SHARED-NOT:    name("nuge")
! SHARED:        acc.loop {
! SHARED:        } seq  unstructured
! PRIVATE:       acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) implicit(true) name("nuge")
! PRIVATE:       acc.loop private({{.*}}) {
! PRIVATE:       } seq  unstructured
! ALL:           acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) implicit(true) name("nuge")
! ALL:           acc.loop vector private({{.*}}) control

! The IV of the seq loop is not private on the loop, while the enclosing seq
! loop (structured) keeps its own private IV.
subroutine seq_nest_inner_goto(a, n, r)
  integer :: n, i, j, k
  real :: a(n, 3, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do j = 1, 3
      !$acc loop seq
      do k = 1, 3
        if (a(i, j, k) .gt. 0.0) goto 50
      end do
50    r(i) = r(i) + j + k
    end do
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPseq_nest_inner_goto
! ALL:         acc.loop gang private({{.*}}) control
! ALL:           name("j")
! ALL:           acc.loop private({{.*}}) control
! SHARED-NOT:      name("k")
! SHARED:          acc.loop {
! PRIVATE:         name("k")
! PRIVATE:         acc.loop private({{.*}}) {
! ALL:             } seq  unstructured
! ALL:           } {{.*}} seq

! Both seq loops are unstructured: neither the outer nor the inner IV is
! privatized by its own loop.
subroutine seq_in_seq(a, n, r)
  integer :: n, j, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop seq
  do j = 1, n
    !$acc loop seq
    do k = 1, 3
      if (a(j, k) .gt. 0.0) goto 50
    end do
50  r(j) = k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPseq_in_seq
! ALL:         name("j")
! ALL:         acc.loop private({{.*}}) control
! SHARED-NOT:    name("k")
! SHARED:        acc.loop {
! PRIVATE:       name("k")
! PRIVATE:       acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

! collapse(2) on an unstructured seq loop: both IVs are shared.
subroutine collapsed(a, n, r)
  integer :: n, i, j, k
  real :: a(n, n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq collapse(2)
    do j = 1, n
      do k = 1, 3
        if (a(i, j, k) .gt. 0.0) goto 50
      end do
    end do
50  r(i) = j + k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPcollapsed
! ALL:         acc.loop gang
! SHARED-NOT:    acc.private
! SHARED:        acc.loop {
! PRIVATE:       acc.loop private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
! ALL:           } collapse([2]) {{.*}} seq  unstructured

! The parent is a compute construct, not a loop: the IV stays private on the
! seq loop.
subroutine parent_is_serial(a, r)
  integer :: k
  real :: a(3), r(1)
  !$acc serial
  !$acc loop seq
  do k = 1, 3
    if (a(k) .gt. 0.0) goto 50
  end do
50 r(1) = k
  !$acc end serial
end subroutine

! ALL-LABEL: func.func @_QPparent_is_serial
! ALL:         acc.serial
! ALL:           acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) implicit(true) name("k")
! ALL:           acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

subroutine parent_is_parallel(a, n, r)
  integer :: n, k
  real :: a(n), r(1)
  !$acc parallel
  !$acc loop seq
  do k = 1, n
    if (a(k) .gt. 0.0) goto 50
  end do
50 r(1) = k
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPparent_is_parallel
! ALL:         acc.parallel
! ALL:           acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) implicit(true) name("k")
! ALL:           acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

! A host `do` between the parent acc.loop and the seq loop does not change
! which loop is the parent.
subroutine host_do_between(a, n, r)
  integer :: n, i, j, k
  real :: a(n, n, 3), r(n, n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    do j = 1, n
      !$acc loop seq
      do k = 1, 3
        if (a(i, j, k) .gt. 0.0) goto 50
      end do
50    r(i, j) = k
    end do
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPhost_do_between
! ALL:         acc.loop gang
! SHARED-NOT:    name("k")
! SHARED:        acc.loop {
! PRIVATE:       name("k")
! PRIVATE:       acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

! A seq loop nested in a collapsed gang loop: the collapsed IVs stay private on
! the gang loop, only the seq IV is shared.
subroutine seq_inner_collapse_outer(a, n, r)
  integer :: n, i, j, k
  real :: a(n, n, 3), r(n, n)
  !$acc parallel
  !$acc loop gang collapse(2)
  do i = 1, n
    do j = 1, n
      !$acc loop seq
      do k = 1, 3
        if (a(i, j, k) .gt. 0.0) goto 50
      end do
50    r(i, j) = k
    end do
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPseq_inner_collapse_outer
! ALL:         acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control
! SHARED-NOT:    name("k")
! SHARED:        acc.loop {
! PRIVATE:       name("k")
! PRIVATE:       acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

! seq applies only to the nvidia device type; for the default device type the
! loop is not seq, so its IV stays private.
subroutine device_type_before_seq(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop device_type(nvidia) seq
    do k = 1, 3
      if (a(i, k) .gt. 0.0) goto 50
    end do
50  r(i) = k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPdevice_type_before_seq
! ALL:         acc.loop gang
! ALL:           name("k")
! ALL:           acc.loop private({{.*}}) {
! ALL:           } seq ([#acc.device_type<nvidia>]) {{.*}} unstructured

! seq is given before device_type, so it applies to the default device type.
subroutine seq_before_device_type(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq device_type(nvidia) collapse(1)
    do k = 1, 3
      if (a(i, k) .gt. 0.0) goto 50
    end do
50  r(i) = k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPseq_before_device_type
! ALL:         acc.loop gang
! SHARED-NOT:    name("k")
! SHARED:        acc.loop {
! PRIVATE:       name("k")
! PRIVATE:       acc.loop private({{.*}}) {
! ALL:           } {{.*}} seq  unstructured

! Only seq loops share their IV; auto may become parallel, so it stays private.
subroutine auto_loop(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop auto
    do k = 1, 3
      if (a(i, k) .gt. 0.0) goto 50
    end do
50  r(i) = k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPauto_loop
! ALL:         acc.loop gang
! ALL:           name("k")
! ALL:           acc.loop private({{.*}}) {
! ALL:           } auto_  unstructured

! An explicit private clause on the IV is honored.
subroutine explicit_private(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq private(k)
    do k = 1, 3
      if (a(i, k) .gt. 0.0) goto 50
    end do
50  r(i) = k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPexplicit_private
! ALL:         acc.loop gang
! ALL:           acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("k")
! ALL:           acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

! Other clauses on the loop are unaffected: the reduction is kept and only the
! IV private disappears.
subroutine reduction_in_seq(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n), s
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    s = 0.0
    !$acc loop seq reduction(+:s)
    do k = 1, 3
      s = s + a(i, k)
      if (s .gt. 10.0) goto 50
    end do
50  r(i) = s + k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPreduction_in_seq
! ALL:         acc.loop gang
! SHARED-NOT:    name("k")
! SHARED:        acc.loop reduction({{.*}}) {
! PRIVATE:       name("k")
! PRIVATE:       acc.loop private({{.*}}) reduction({{.*}}) {
! ALL:           } seq  unstructured

! Two sibling seq loops reusing an integer(8) IV: neither gets a private.
subroutine siblings_i8(a, n, r)
  integer :: n, i
  integer(8) :: k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, 3
      if (a(i, k) .gt. 0.0) goto 50
    end do
50  r(i) = k
    !$acc loop seq
    do k = 1, 3
      if (a(i, k) .gt. 1.0) goto 60
    end do
60  r(i) = r(i) + k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPsiblings_i8
! ALL:         acc.loop gang
! SHARED-NOT:    name("k")
! SHARED:        acc.loop {
! SHARED:        } seq  unstructured
! SHARED:        acc.loop {
! SHARED:        } seq  unstructured
! PRIVATE:       acc.private varPtr({{.*}}) recipe(@privatization_ref_i64) implicit(true) name("k")
! PRIVATE:       acc.loop private({{.*}} : !fir.ref<i64>) {
! PRIVATE:       } seq  unstructured
! PRIVATE:       acc.private varPtr({{.*}}) recipe(@privatization_ref_i64) implicit(true) name("k")
! PRIVATE:       acc.loop private({{.*}} : !fir.ref<i64>) {
! PRIVATE:       } seq  unstructured

! No enclosing loop or compute construct at all: nothing to share with.
subroutine orphan(a, r)
  integer :: k
  real :: a(3), r(1)
  !$acc loop seq
  do k = 1, 3
    if (a(k) .gt. 0.0) goto 50
  end do
50 r(1) = k
end subroutine

! ALL-LABEL: func.func @_QPorphan
! ALL:         acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) implicit(true) name("k")
! ALL:         acc.loop private({{.*}}) {
! ALL:         } seq  unstructured

! exit makes the loop unstructured as well.
subroutine exit_stmt(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, 3
      if (a(i, k) .gt. 0.0) exit
    end do
    r(i) = k
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPexit_stmt
! ALL:         acc.loop gang
! SHARED-NOT:    name("k")
! SHARED:        acc.loop {
! PRIVATE:       name("k")
! PRIVATE:       acc.loop private({{.*}}) {
! ALL:           } seq  unstructured

! Structured seq loops (with control operands) are not affected.
subroutine structured_seq(a, n, r)
  integer :: n, i, k
  real :: a(n, 3), r(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, 3
      r(i) = r(i) + a(i, k)
    end do
  end do
  !$acc end parallel
end subroutine

! ALL-LABEL: func.func @_QPstructured_seq
! ALL:         acc.loop gang
! ALL:           name("k")
! ALL:           acc.loop private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! ALL:           } {{.*}} seq

! A loop without a seq clause that is only implicitly sequential (inside a
! serial construct) is not treated as seq.
subroutine implicit_seq_in_serial(a, r)
  integer :: k
  real :: a(3), r(1)
  !$acc serial
  !$acc loop
  do k = 1, 3
    if (a(k) .gt. 0.0) goto 50
  end do
50 r(1) = k
  !$acc end serial
end subroutine

! ALL-LABEL: func.func @_QPimplicit_seq_in_serial
! ALL:         name("k")
! ALL:         acc.loop private({{.*}}) {
! ALL:         } seq  unstructured
