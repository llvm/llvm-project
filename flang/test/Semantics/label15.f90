! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

!CHECK-NOT: error:
module mm
   interface
      module subroutine m(n)
      end
   end interface
end module mm

program p
   use mm
20 print*, 'p'
21 call p1
22 call p2
23 f0 = f(0); print '(f5.1)', f0
24 f1 = f(1); print '(f5.1)', f1
25 call s(0); call s(1)
26 call m(0); call m(1)
27 if (.false.) goto 29
28 print*, 'px'
contains
   subroutine p1
      print*, 'p1'
      goto 29
29 end subroutine p1
   subroutine p2
      print*, 'p2'
      goto 29
29 end subroutine p2
29 end

function f(n)
   print*, 'f'
31 call f1
32 call f2
   f = 30.
   if (n == 0) goto 39
   f = f + 3.
   print*, 'fx'
contains
   subroutine f1
      print*, 'f1'
      goto 39
39 end subroutine f1
   subroutine f2
      print*, 'f2'
      goto 39
39 end subroutine f2
39 end

subroutine s(n)
   print*, 's'
41 call s1
42 call s2
43 call s3
   if (n == 0) goto 49
   print*, 'sx'
contains
   subroutine s1
      print*, 's1'
      goto 49
49 end subroutine s1
   subroutine s2
      print*, 's2'
      goto 49
49 end subroutine s2
   subroutine s3
      print*, 's3'
      goto 49
49 end subroutine s3
49 end

submodule(mm) mm1
contains
   module procedure m
      print*, 'm'
   50 call m1
   51 call m2
      if (n == 0) goto 59
      print*, 'mx'
   contains
      subroutine m1
         print*, 'm1'
         goto 59
   59 end subroutine m1
      subroutine m2
         print*, 'm2'
         goto 59
   59 end subroutine m2
   59 end procedure m
end submodule mm1
