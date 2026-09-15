! RUN: %flang_fc1 -fsyntax-only %s 2>&1

      program p
      integer :: y, x
      y = 3
      x = y
     /* 2
      print *, x
      print *, 'tail */ text'
      end
