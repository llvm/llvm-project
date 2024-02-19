! Test "nofinalizationneeded" is set to false for derived type
! containing polymorphic allocatable ultimate components.
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

  type :: t_base
  end type
  type :: t_container_not_polymorphic
     type(t_base), allocatable :: comp
  end type
  type :: t_container
     class(t_base), allocatable :: comp
  end type
  type, extends(t_container) :: t_container_extension
  end type
  type :: t_container_wrapper
    type(t_container_extension) :: wrapper
  end type
end
! CHECK: .dt.t_container, SAVE, TARGET (CompilerCreated, ReadOnly): {{.*}}noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=0_1)
! CHECK: .dt.t_container_extension, SAVE, TARGET (CompilerCreated, ReadOnly): {{.*}}noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=0_1)
! CHECK: .dt.t_container_not_polymorphic, SAVE, TARGET (CompilerCreated, ReadOnly): {{.*}}noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=1_1)
! CHECK: .dt.t_container_wrapper, SAVE, TARGET (CompilerCreated, ReadOnly): {{.*}}noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=0_1)
