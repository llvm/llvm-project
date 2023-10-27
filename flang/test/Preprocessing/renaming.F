! RUN: %flang -E %s | FileCheck %s
! CHECK: ((1)*10000+(11)*100)
! Ensure that a keyword-like macro can be used to rename a
! function-like macro.
#define TO_VERSION2(MAJOR, MINOR) ((MAJOR) * 10000 + (MINOR) * 100)
#define TO_VERSION TO_VERSION2
      print *, TO_VERSION(1,11)
      end
