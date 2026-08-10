! RUN: %flang_fc1 -fsyntax-only -cpp %s 2>&1
#define UNITY(k) 1_ ## k
PROGRAM REPRODUCER
WRITE(*,*) UNITY(4)
END PROGRAM REPRODUCER