! RUN: bbc -fopenmp --strict-fir-volatile-verifier %s -o - | FileCheck %s

! Verify that VOLATILE and ASYNCHRONOUS attributes added to a host-associated
! name are reflected in the declaration inside the internal procedure without
! changing the declaration in the host procedure.

subroutine local_volatile_global
  integer :: n = 0
  call inner
contains
  subroutine inner
    volatile :: n
    do while (n == 0)
    end do
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPlocal_volatile_global()
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QFlocal_volatile_globalEn) : !fir.ref<i32>
! CHECK-NOT:     fir.volatile_cast %[[ADDR]]
! CHECK:         %[[HOST_DECLARE:.*]]:2 = hlfir.declare %[[ADDR]] uniq_name("_QFlocal_volatile_globalEn") fortran_attrs<internal_assoc> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK-LABEL: func.func private @_QFlocal_volatile_globalPinner()
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QFlocal_volatile_globalEn) : !fir.ref<i32>
! CHECK:         %[[VOLATILE_ADDR:.*]] = fir.volatile_cast %[[ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         %[[INNER_DECLARE:.*]]:2 = hlfir.declare %[[VOLATILE_ADDR]] uniq_name("_QFlocal_volatile_globalEn") fortran_attrs<volatile> : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:         fir.load %[[INNER_DECLARE]]#0 : !fir.ref<i32, volatile>

subroutine local_volatile_tuple
  integer :: n
  n = 0
  call inner
contains
  subroutine inner
    volatile :: n
    do while (n == 0)
    end do
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPlocal_volatile_tuple()
! CHECK:         %[[HOST_ADDR:.*]] = fir.alloca i32
! CHECK-NOT:     fir.volatile_cast %[[HOST_ADDR]]
! CHECK:         %[[HOST_DECLARE:.*]]:2 = hlfir.declare %[[HOST_ADDR]] uniq_name("_QFlocal_volatile_tupleEn") fortran_attrs<internal_assoc> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         fir.call @_QFlocal_volatile_tuplePinner({{.*}}) {{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()

! CHECK-LABEL: func.func private @_QFlocal_volatile_tuplePinner(
! CHECK-SAME:    %[[HOST_TUPLE:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
! CHECK:         %[[COORD:.*]] = fir.coordinate_of %[[HOST_TUPLE]], {{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:         %[[TUPLE_ADDR:.*]] = fir.load %[[COORD]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:         %[[VOLATILE_ADDR:.*]] = fir.volatile_cast %[[TUPLE_ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         %[[INNER_DECLARE:.*]]:2 = hlfir.declare %[[VOLATILE_ADDR]] uniq_name("_QFlocal_volatile_tupleEn") fortran_attrs<volatile, host_assoc> : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:         fir.load %[[INNER_DECLARE]]#0 : !fir.ref<i32, volatile>

subroutine local_asynchronous_tuple
  integer :: n
  n = 0
  call inner
contains
  subroutine inner
    asynchronous :: n
    n = n + 1
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFlocal_asynchronous_tuplePinner(
! CHECK:         %[[INNER_DECLARE:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFlocal_asynchronous_tupleEn") fortran_attrs<asynchronous, host_assoc>

! Verify that a local ASYNCHRONOUS attribute is retained when an OpenMP target
! region rebinds the current scope's host-associated symbol.
subroutine local_asynchronous_omp_target
  integer :: n
  n = 0
  call inner
contains
  subroutine inner
    asynchronous :: n
    !$omp target map(tofrom:n)
      n = n + 1
    !$omp end target
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFlocal_asynchronous_omp_targetPinner(
! CHECK:         %[[OMP_HOST_DECLARE:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFlocal_asynchronous_omp_targetEn") fortran_attrs<asynchronous, host_assoc>
! CHECK:         omp.target {{.*}}map_entries({{.*}} -> %[[OMP_TARGET_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[OMP_TARGET_DECLARE:.*]]:2 = hlfir.declare %[[OMP_TARGET_ARG]] uniq_name("_QFlocal_asynchronous_omp_targetEn") fortran_attrs<asynchronous>
! CHECK:           %{{.*}} = fir.load %[[OMP_TARGET_DECLARE]]#0 : !fir.ref<i32>

subroutine blk
  integer :: n = 0
  block
    volatile :: n
    do while (n == 0)
    end do
  end block
end subroutine

! CHECK-LABEL: func.func @_QPblk()
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QFblkEn) : !fir.ref<i32>
! CHECK:         %[[HOST_DECLARE:.*]]:2 = hlfir.declare %[[ADDR]] uniq_name("_QFblkEn") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VOLATILE_ADDR:.*]] = fir.volatile_cast %[[HOST_DECLARE]]#0 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         %[[BLOCK_DECLARE:.*]]:2 = hlfir.declare %[[VOLATILE_ADDR]] uniq_name("_QFblkEn") fortran_attrs<volatile> : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:         fir.load %[[BLOCK_DECLARE]]#0 : !fir.ref<i32, volatile>

module m
  integer :: n = 0
end module

subroutine poll
  use m
  volatile :: n
  do while (n == 0)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPpoll()
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMmEn) : !fir.ref<i32>
! CHECK:         %[[VOLATILE_ADDR:.*]] = fir.volatile_cast %[[ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         %[[DECLARE:.*]]:2 = hlfir.declare %[[VOLATILE_ADDR]] uniq_name("_QMmEn") fortran_attrs<volatile> : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:         fir.load %[[DECLARE]]#0 : !fir.ref<i32, volatile>

module nested_assoc_m
  integer, target :: n = 0
end module

subroutine nested_use_host
  use nested_assoc_m, only : n
  call plain_access
  call volatile_poll
contains
  subroutine plain_access
    n = n + 1
  end subroutine

  subroutine volatile_poll
    volatile :: n
    do while (n == 0)
    end do
  end subroutine
end subroutine

! The sibling procedure retains the module variable's TARGET attribute and
! must not inherit VOLATILE from volatile_poll.
! CHECK-LABEL: func.func private @_QFnested_use_hostPplain_access()
! CHECK:         %[[PLAIN_ADDR:.*]] = fir.address_of(@_QMnested_assoc_mEn) : !fir.ref<i32>
! CHECK-NOT:     fir.volatile_cast %[[PLAIN_ADDR]]
! CHECK:         %[[PLAIN_DECLARE:.*]]:2 = hlfir.declare %[[PLAIN_ADDR]] uniq_name("_QMnested_assoc_mEn") fortran_attrs<target> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! The local VOLATILE attribute is combined with TARGET from the ultimate
! module symbol in volatile_poll.
! CHECK-LABEL: func.func private @_QFnested_use_hostPvolatile_poll()
! CHECK:         %[[POLL_ADDR:.*]] = fir.address_of(@_QMnested_assoc_mEn) : !fir.ref<i32>
! CHECK:         %[[POLL_VOLATILE_ADDR:.*]] = fir.volatile_cast %[[POLL_ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         %[[POLL_DECLARE:.*]]:2 = hlfir.declare %[[POLL_VOLATILE_ADDR]] uniq_name("_QMnested_assoc_mEn") fortran_attrs<target, volatile> : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:         fir.load %[[POLL_DECLARE]]#0 : !fir.ref<i32, volatile>

! Verify that the local name of a renamed USE association is used to find the
! VOLATILE attribute, while the declaration still refers to the ultimate
! module symbol.
module renamed_assoc_m
  integer, target :: source_counter = 0
end module

subroutine renamed_use_host(limit)
  use renamed_assoc_m, only : local_counter => source_counter
  integer, intent(in) :: limit
  call plain_renamed_access
  call volatile_renamed_poll
contains
  subroutine plain_renamed_access
    local_counter = local_counter + 1
  end subroutine

  subroutine volatile_renamed_poll
    volatile :: local_counter
    do while (local_counter < limit)
      local_counter = local_counter + 1
    end do
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFrenamed_use_hostPplain_renamed_access(
! CHECK:         %[[RENAMED_PLAIN_ADDR:.*]] = fir.address_of(@_QMrenamed_assoc_mEsource_counter) : !fir.ref<i32>
! CHECK-NOT:     fir.volatile_cast %[[RENAMED_PLAIN_ADDR]]
! CHECK:         hlfir.declare %[[RENAMED_PLAIN_ADDR]] uniq_name("_QMrenamed_assoc_mEsource_counter") fortran_attrs<target>

! CHECK-LABEL: func.func private @_QFrenamed_use_hostPvolatile_renamed_poll(
! CHECK:         %[[RENAMED_ADDR:.*]] = fir.address_of(@_QMrenamed_assoc_mEsource_counter) : !fir.ref<i32>
! CHECK:         %[[RENAMED_VOLATILE_ADDR:.*]] = fir.volatile_cast %[[RENAMED_ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         hlfir.declare %[[RENAMED_VOLATILE_ADDR]] uniq_name("_QMrenamed_assoc_mEsource_counter") fortran_attrs<target, volatile>

! Verify a local ASYNCHRONOUS attribute on a host-associated dummy argument.
! The dummy's INTENT attribute must be preserved, and the sibling procedure
! must continue to use a non-ASYNCHRONOUS declaration.
subroutine dummy_assoc_host(counter, limit)
  integer, intent(inout) :: counter
  integer, intent(in) :: limit
  call plain_dummy_access
  call asynchronous_dummy_update
contains
  subroutine plain_dummy_access
    counter = counter + 1
  end subroutine

  subroutine asynchronous_dummy_update
    asynchronous :: counter
    if (counter < limit) then
      counter = counter + 1
    endif
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFdummy_assoc_hostPplain_dummy_access(
! CHECK:         hlfir.declare {{.*}} uniq_name("_QFdummy_assoc_hostEcounter") fortran_attrs<intent_inout, host_assoc>

! CHECK-LABEL: func.func private @_QFdummy_assoc_hostPasynchronous_dummy_update(
! CHECK:         hlfir.declare {{.*}} uniq_name("_QFdummy_assoc_hostEcounter") fortran_attrs<asynchronous, intent_inout, host_assoc>

! An internal subprogram cannot itself contain another internal subprogram.
! Use a module variable in a module procedure's internal procedures to cover
! two association levels: module scope to module procedure to internal
! procedure.
module multilevel_assoc_m
  integer, target :: state = 0
contains
  subroutine multilevel_host(limit)
    integer, intent(in) :: limit
    call plain_leaf
    call volatile_leaf
  contains
    subroutine plain_leaf
      state = state + 1
    end subroutine

    subroutine volatile_leaf
      volatile :: state
      do while (state < limit)
        state = state + 1
      end do
    end subroutine
  end subroutine
end module

! CHECK-LABEL: func.func{{.*}}@{{.*}}multilevel_hostPplain_leaf(
! CHECK:         %[[MULTILEVEL_PLAIN_ADDR:.*]] = fir.address_of(@_QMmultilevel_assoc_mEstate) : !fir.ref<i32>
! CHECK-NOT:     fir.volatile_cast %[[MULTILEVEL_PLAIN_ADDR]]
! CHECK:         hlfir.declare %[[MULTILEVEL_PLAIN_ADDR]] uniq_name("_QMmultilevel_assoc_mEstate") fortran_attrs<target>

! CHECK-LABEL: func.func{{.*}}@{{.*}}multilevel_hostPvolatile_leaf(
! CHECK:         %[[MULTILEVEL_ADDR:.*]] = fir.address_of(@_QMmultilevel_assoc_mEstate) : !fir.ref<i32>
! CHECK:         %[[MULTILEVEL_VOLATILE_ADDR:.*]] = fir.volatile_cast %[[MULTILEVEL_ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         hlfir.declare %[[MULTILEVEL_VOLATILE_ADDR]] uniq_name("_QMmultilevel_assoc_mEstate") fortran_attrs<target, volatile>

! Verify that a BLOCK inside an internal procedure can add VOLATILE to an
! entity that is already host-associated with the internal procedure. Uses
! before and after the BLOCK retain the outer non-VOLATILE binding.
subroutine internal_block_host
  integer :: state
  state = 0
  call worker
contains
  subroutine worker
    asynchronous :: state
    state = state + 1
    block
      volatile :: state
      do while (state == 0)
      end do
    end block
    state = state + 1
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFinternal_block_hostPworker(
! CHECK:         %[[INTERNAL_BLOCK_HOST_DECLARE:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFinternal_block_hostEstate") fortran_attrs<asynchronous, host_assoc>
! CHECK:         %[[INTERNAL_BLOCK_VOLATILE_ADDR:.*]] = fir.volatile_cast %[[INTERNAL_BLOCK_HOST_DECLARE]]#0
! CHECK:         %[[INTERNAL_BLOCK_DECLARE:.*]]:2 = hlfir.declare %[[INTERNAL_BLOCK_VOLATILE_ADDR]] uniq_name("_QFinternal_block_hostEstate") fortran_attrs<asynchronous, volatile>
! CHECK:         fir.load %[[INTERNAL_BLOCK_DECLARE]]#0 : !fir.ref<i32, volatile>

! Exercise local VOLATILE on several kinds of host-associated entities. This
! covers the shape and type-parameter metadata of arrays and characters, and
! descriptor-based pointer and allocatable entities.
subroutine object_kinds_host
  integer :: values(4)
  character(12) :: message
  integer, pointer :: cursor
  integer, allocatable :: scratch(:)
  values = [1, 2, 3, 4]
  message = "initial"
  allocate(cursor, scratch(4))
  cursor = 1
  scratch = values
  call plain_objects
  call volatile_objects
  deallocate(cursor, scratch)
contains
  subroutine plain_objects
    values(1) = values(1) + cursor
    message(1:1) = "p"
    scratch(1) = values(1)
  end subroutine

  subroutine volatile_objects
    volatile :: values, message, cursor, scratch
    values(2) = values(1) + cursor
    message(1:1) = "v"
    scratch(2) = values(2)
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFobject_kinds_hostPplain_objects(
! CHECK-NOT:     fir.volatile_cast
! CHECK:         return

! CHECK-LABEL: func.func private @_QFobject_kinds_hostPvolatile_objects(
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFobject_kinds_hostEvalues") fortran_attrs<volatile, host_assoc>
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFobject_kinds_hostEmessage") fortran_attrs<volatile, host_assoc>
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFobject_kinds_hostEcursor") fortran_attrs<pointer, volatile, host_assoc>
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFobject_kinds_hostEscratch") fortran_attrs<allocatable, volatile, host_assoc>

! Verify local VOLATILE on a host-associated COMMON block object. The sibling
! procedure provides a non-VOLATILE view of the same common storage.
subroutine common_assoc_host(limit)
  integer :: shared_state, generation
  integer, intent(in) :: limit
  common /volatile_common_state/ shared_state, generation
  call plain_common_update
  call volatile_common_poll
contains
  subroutine plain_common_update
    generation = generation + 1
  end subroutine

  subroutine volatile_common_poll
    volatile :: shared_state
    do while (shared_state < limit)
      shared_state = shared_state + 1
    end do
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFcommon_assoc_hostPplain_common_update(
! CHECK-NOT:     fir.volatile_cast
! CHECK:         hlfir.declare {{.*}} uniq_name("_QFcommon_assoc_hostEgeneration")

! CHECK-LABEL: func.func private @_QFcommon_assoc_hostPvolatile_common_poll(
! CHECK:         %[[COMMON_VOLATILE_ADDR:.*]] = fir.volatile_cast {{.*}}
! CHECK:         hlfir.declare %[[COMMON_VOLATILE_ADDR]] {{.*}}uniq_name("_QFcommon_assoc_hostEshared_state") fortran_attrs<volatile>

! Verify local VOLATILE on an equivalenced host entity. Accessing another
! member of the equivalence set in the sibling also checks that VOLATILE stays
! attached to the local associated name rather than to the shared storage.
subroutine equivalence_assoc_host
  integer :: lanes(2), alias
  equivalence(lanes(2), alias)
  lanes = [1, 2]
  call plain_equivalence_access
  call volatile_equivalence_access
contains
  subroutine plain_equivalence_access
    lanes(1) = lanes(1) + 1
  end subroutine

  subroutine volatile_equivalence_access
    volatile :: alias
    alias = alias + lanes(1)
  end subroutine
end subroutine

! CHECK-LABEL: func.func private @_QFequivalence_assoc_hostPplain_equivalence_access(
! CHECK-NOT:     fir.volatile_cast
! CHECK:         hlfir.declare {{.*}} uniq_name("_QFequivalence_assoc_hostElanes")

! CHECK-LABEL: func.func private @_QFequivalence_assoc_hostPvolatile_equivalence_access(
! CHECK:         %[[EQUIV_VOLATILE_ADDR:.*]] = fir.volatile_cast {{.*}}
! CHECK:         hlfir.declare %[[EQUIV_VOLATILE_ADDR]] {{.*}}uniq_name("_QFequivalence_assoc_hostEalias") fortran_attrs<volatile, host_assoc>

! Verify an attribute added to a name associated with an ancestor module in a
! submodule procedure.
module ancestor_assoc_m
  integer, target :: ancestor_state = 0
  interface
    module subroutine ancestor_poll(limit)
      integer, intent(in) :: limit
    end subroutine
  end interface
end module

submodule(ancestor_assoc_m) ancestor_assoc_impl
contains
  module procedure ancestor_poll
    volatile :: ancestor_state
    do while (ancestor_state < limit)
      ancestor_state = ancestor_state + 1
    end do
  end procedure
end submodule

! CHECK-LABEL: func.func @_QMancestor_assoc_mPancestor_poll(
! CHECK:         %[[ANCESTOR_ADDR:.*]] = fir.address_of(@_QMancestor_assoc_mEancestor_state) : !fir.ref<i32>
! CHECK:         %[[ANCESTOR_VOLATILE_ADDR:.*]] = fir.volatile_cast %[[ANCESTOR_ADDR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:         hlfir.declare %[[ANCESTOR_VOLATILE_ADDR]] uniq_name("_QMancestor_assoc_mEancestor_state") fortran_attrs<target, volatile>

! Verify attributes added in a submodule scope remain visible in an internal
! procedure nested inside a module procedure of that submodule.
module nested_submodule_assoc_m
  integer :: n = 0, k = 0
  interface
    module subroutine nested_submodule_poll
    end subroutine
  end interface
end module

submodule(nested_submodule_assoc_m) nested_submodule_assoc_impl
  volatile :: n
  asynchronous :: k
contains
  module subroutine nested_submodule_poll
    call inner
  contains
    subroutine inner
      n = n + 1
      k = k + 1
    end subroutine
  end subroutine
end submodule

! CHECK-LABEL: func.func private @{{.*}}nested_submodule_pollPinner(
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QMnested_submodule_assoc_mEn") fortran_attrs<volatile>
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QMnested_submodule_assoc_mEk") fortran_attrs<asynchronous>

! Verify attributes added to use-associated names in a submodule scope remain
! visible in a module procedure contained in that submodule.
module submodule_use_source_m
  integer :: use_n = 0, use_k = 0
end module

module submodule_use_parent_m
  interface
    module subroutine submodule_use_poll
    end subroutine
  end interface
end module

submodule(submodule_use_parent_m) submodule_use_assoc_impl
  use submodule_use_source_m, only: use_n, use_k
  volatile :: use_n
  asynchronous :: use_k
contains
  module subroutine submodule_use_poll
    use_n = use_n + 1
    use_k = use_k + 1
  end subroutine
end submodule

! CHECK-LABEL: func.func @_QMsubmodule_use_parent_mPsubmodule_use_poll(
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QMsubmodule_use_source_mEuse_n") fortran_attrs<volatile>
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QMsubmodule_use_source_mEuse_k") fortran_attrs<asynchronous>
