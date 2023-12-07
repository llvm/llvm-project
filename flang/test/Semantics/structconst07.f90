! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
type :: hasPointer
  class(*), pointer :: sp
end type
type(hasPointer) hp
!CHECK: hp=haspointer(sp=NULL())
hp = hasPointer()
end

