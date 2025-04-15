! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s
! Ensure that EOSHIFT's ARRAY= argument and result can be CLASS(*).
! CHECK-NOT: error:
! CHECK: warning: Source of TRANSFER is polymorphic
! CHECK: warning: Mold of TRANSFER is polymorphic
program p
  type base
    integer j
  end type
  type, extends(base) :: extended
    integer k
  end type
  class(base), allocatable :: polyArray(:,:,:)
  class(*), allocatable :: unlimited(:)
  allocate(polyArray, source=reshape([(extended(n,n-1),n=1,8)],[2,2,2]))
  allocate(unlimited, source=[(base(9),n=1,16)])
  select type (x => eoshift(transfer(polyArray, unlimited), -4, base(-1)))
    type is (base); print *, 'base', x
    type is (extended); print *, 'extended?', x
    class default; print *, 'class default??'
  end select
end
