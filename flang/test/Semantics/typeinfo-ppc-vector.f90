!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
!REQUIRES: target=powerpc{{.*}}
! Test that PowerPC intrinsic vector types have isVectorType=1

subroutine test
  implicit none
  type dt1
    vector(integer(4)) :: v
  end type
  type dt2
    vector(real(8)), allocatable :: v
  end type
end

!CHECK: .dt.__builtin_ppc_intrinsic_vector{{.*}}, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype({{.*}}isvectortype=1_1)
!CHECK: .dt.dt1, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.dt1,sizeinbytes=16_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.dt1,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1,isvectortype=0_1)
!CHECK: .dt.dt2, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.dt2,sizeinbytes=24_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.dt2,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=1_1,nodefinedassignment=1_1,isvectortype=0_1)

