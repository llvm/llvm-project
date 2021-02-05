! Verify that reading from stdin works as expected

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! TODO: Add support for `flang-new -`
! Currently `bin/flang-new  -E -` defaults to `-x c` and e.g. F90 is not allowed
! in `-x <input-type>` (see `clang::driver::types::canTypeBeUserSpecified` in
! Types.cpp)

!---------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!---------------------------------------
! Test `-E` - for the corresponding frontend actions the driver relies on the prescanner API to handle file I/O
! RUN: cat %s | flang-new -fc1 -E | FileCheck %s --check-prefix=PP-NOT-DEFINED
! RUN: cat %s | flang-new -fc1 -DNEW -E | FileCheck %s --check-prefix=PP-DEFINED

! Test `-test-io` - for the corresponding frontend action (`InputOutputTestAction`) the driver handles the file I/O on its own
! the corresponding action (`PrintPreprocessedAction`)
! RUN: cat %s | flang-new -fc1 -test-io | FileCheck %s --check-prefix=IO --match-full-lines
! RUN: cat %s | flang-new -fc1 -DNEW -test-io | FileCheck %s --check-prefix=IO --match-full-lines

!-------------------------
! EXPECTED OUTPUT for `-E`
!-------------------------
! PP-NOT-DEFINED: program b
! PP-DEFINED: program a

!-------------------------------
! EXPECTED OUTPUT for `-test-io`
!-------------------------------
! IO: #ifdef NEW
! IO-NEXT:   Program A
! IO-NEXT: #else
! IO-NEXT:   Program B
! IO-NEXT: #endif

#ifdef NEW
  Program A
#else
  Program B
#endif
