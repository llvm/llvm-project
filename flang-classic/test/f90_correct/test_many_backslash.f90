! RUN: not %flang -c -fbackslash %s 2>&1 | FileCheck %s --check-prefix=CHECK-WITH-BSLASH-MSG1
! RUN: not %flang -c -fbackslash %s 2>&1 | FileCheck %s --check-prefix=CHECK-MSG2
! RUN: not %flang -c -fno-backslash %s 2>&1 | FileCheck %s --check-prefix=CHECK-MSG2

program test1
   print *, '\'
end program test1
// CHECK-WITH-BSLASH-MSG1: F90-S-0601-Unmatched quote - backslash treated as escape. Try recompiling with -fno-backslash.

program test2
   print *, '\\'
end program test2

program test3
   print *, '\\\'
end program test3
// CHECK-WITH-BSLASH-MSG1: F90-S-0601-Unmatched quote - backslash treated as escape. Try recompiling with -fno-backslash.

program test4
   print *, 'Hello world\'
end program test4
// CHECK-WITH-BSLASH-MSG1: F90-S-0601-Unmatched quote - backslash treated as escape. Try recompiling with -fno-backslash.

program test5
   print *, 'Hello world\\'
end program test5

program test6
   print *, 'Hello world\\\'
end program test6
// CHECK-WITH-BSLASH-MSG1: F90-S-0601-Unmatched quote - backslash treated as escape. Try recompiling with -fno-backslash.

program test7
   print *, 'Hello \'world'
end program test7

program test8
   print *, 'Hello \\'world'
end program test8
// CHECK-MSG2: F90-S-0026-Unmatched quote

program test9
   print *, 'Hello world
end program test9
// CHECK-MSG2: F90-S-0026-Unmatched quote
