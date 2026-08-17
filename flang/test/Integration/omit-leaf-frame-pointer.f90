! Test to check the options -momit-leaf-frame-pointer and -mno-omit-leaf-frame-pointer

! REQUIRES: aarch64-registered-target, x86-registered-target

! AArch64 differs from x86_64: useLeafFramePointerForTargetByDefault is false, so -fno-omit-frame-pointer alone already gives "non-leaf-no-reserve" on AArch64.

! RUN: %flang --target=x86_64-unknown-linux-gnu -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ALL
! RUN: %flang --target=x86_64-unknown-linux-gnu -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ALL
! RUN: %flang --target=aarch64-none-none -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ALL

! RUN: %flang --target=x86_64-unknown-linux-gnu -fno-omit-frame-pointer -momit-leaf-frame-pointer -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONLEAF
! RUN: %flang --target=aarch64-none-none -fno-omit-frame-pointer -momit-leaf-frame-pointer -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONLEAF


! CHECK-ALL: define void @_QQmain() #[[ATTR:[0-9]+]]
! CHECK-ALL: define internal i32 @_QFPleaf(i32 %{{.*}}, i32 %{{.*}}) #[[ATTR]]
! CHECK-ALL: attributes #[[ATTR]] = {{{.*}}"frame-pointer"="all"{{.*}}}
! CHECK-ALL: !{{[0-9]+}} = !{i32 7, !"frame-pointer", i32 2}

! CHECK-NONLEAF: define void @_QQmain() #[[ATTR:[0-9]+]]
! CHECK-NONLEAF: define internal i32 @_QFPleaf(i32 %{{.*}}, i32 %{{.*}}) #[[ATTR]]
! CHECK-NONLEAF: attributes #[[ATTR]] = {{{.*}}"frame-pointer"="non-leaf-no-reserve"{{.*}}}
! CHECK-NONLEAF: !{{[0-9]+}} = !{i32 7, !"frame-pointer", i32 4}

program test
   implicit none
   integer :: result
   result = leaf(3, 4)
   print *, result
contains
   integer function leaf(a, b)
      implicit none
      integer, value :: a, b
      integer :: temp
      temp = a + b
      leaf = temp
   end function leaf
end program test

