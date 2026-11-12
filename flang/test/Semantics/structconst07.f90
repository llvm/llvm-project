! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
type :: hasPointer
  class(*), pointer :: sp
end type
type :: hasAllocatable
  class(*), allocatable :: sa
end type
type(hasPointer) hp
type(hasAllocatable) ha
!CHECK: hp=haspointer(sp=NULL())
hp = hasPointer()
!CHECK: ha=hasallocatable(sa=NULL())
ha = hasAllocatable()
end

