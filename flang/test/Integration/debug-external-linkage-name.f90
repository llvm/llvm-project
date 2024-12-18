! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! Test that correct linkage name is generated in the debug info.
subroutine sub(a)
  integer :: a
  return a+1
end

!CHECK: !DISubprogram(name: "sub", linkageName: "sub_"{{.*}})

