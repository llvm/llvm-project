!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
! Regression test for a crash (llvm-project/issues/79590)
module m
  type t(k1)
     integer,kind :: k1
  end type
  type s(l1)
     integer,len :: l1
     type(t(3)) :: t1
  end type
end module

!CHECK: Module scope: m size=0 alignment=1 sourceRange=113 bytes
!CHECK: .c.s, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(component) shape: 0_8:0_8 init:[component::component(name=.n.t1,genre=1_1,category=5_1,kind=0_1,rank=0_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),lenvalue=NULL(),bounds=NULL(),initialization=NULL())]
!CHECK: .dt.s, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.s,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=.lpk.s,component=.c.s,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
!CHECK: .lpk.s, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: INTEGER(1) shape: 0_8:0_8 init:[INTEGER(1)::4_1]
!CHECK: .n.s, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: CHARACTER(1_8,1) init:"s"
!CHECK: .n.t1, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: CHARACTER(2_8,1) init:"t1"
!CHECK: s, PUBLIC: DerivedType components: t1
!CHECK: t, PUBLIC: DerivedType
!CHECK: DerivedType scope: t sourceRange=27 bytes
!CHECK: k1: TypeParam type:INTEGER(4) Kind
!CHECK: DerivedType scope: s size=0 alignment=1 sourceRange=43 bytes
!CHECK: l1: TypeParam type:INTEGER(4) Len
!CHECK: t1: ObjectEntity type: TYPE(t(k1=3_4))
