! Tests for the `-save-temps` flag. As `flang` does not implement `-fc1as` (i.e. a driver for the integrated assembler), we need to
! use `-fno-integrated-as` here.

! UNSUPPORTED: system-windows

!--------------------------
! Basic case: `-save-temps`
!--------------------------
! RUN: %flang -save-temps -fno-integrated-as %s -### 2>&1 | FileCheck %s
! CHECK: "-o" "save-temps.i"
! CHECK-NEXT: "-o" "save-temps.bc"
! CHECK-NEXT: "-o" "save-temps.s"
! CHECK-NEXT: "-o" "save-temps.o"
! CHECK-NEXT: "-o" "a.out"

!--------------------------
! `-save-temps=cwd`
!--------------------------
! This should work the same as -save-temps above

! RUN: %flang -save-temps=cwd -fno-integrated-as  %s -### 2>&1 | FileCheck %s -check-prefix=CWD
! CWD: "-o" "save-temps.i"
! CWD-NEXT: "-o" "save-temps.bc"
! CWD-NEXT: "-o" "save-temps.s"
! CWD-NEXT: "-o" "save-temps.o"
! CWD-NEXT: "-o" "a.out"

!--------------------------
! `-save-temps=obj`
!--------------------------
! Check that temp files are saved in the same directory as the output file
! regardless of whether -o is specified.

! RUN: %flang -save-temps=obj -fno-integrated-as -o obj/dir/a.out %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-OBJ
! CHECK-OBJ: "-o" "obj/dir/save-temps.i"
! CHECK-OBJ-NEXT: "-o" "obj/dir/save-temps.bc"
! CHECK-OBJ-NEXT: "-o" "obj/dir/save-temps.s"
! CHECK-OBJ-NEXT: "-o" "obj/dir/save-temps.o"
! CHECK-OBJ-NEXT: "-o" "obj/dir/a.out"

! RUN: %flang -save-temps=obj -fno-integrated-as %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-OBJ-NOO
! CHECK-OBJ-NOO: "-o" "save-temps.i"
! CHECK-OBJ-NOO-NEXT: "-o" "save-temps.bc"
! CHECK-OBJ-NOO-NEXT: "-o" "save-temps.s"
! CHECK-OBJ-NOO-NEXT: "-o" "save-temps.o"
! CHECK-OBJ-NOO-NEXT: "-o" "a.out"

!--------------------------
! `-S` without `-save-temps`
!--------------------------
! Check for a single `flang -fc1` invocation when NOT using -save-temps.
! RUN: %flang -S %s -### 2>&1 | FileCheck %s -check-prefix=NO-TEMPS
! NO-TEMPS: "-fc1"
! NO-TEMPS-SAME: "-S"
! NO-TEMPS-SAME: "-o" "save-temps.s"
