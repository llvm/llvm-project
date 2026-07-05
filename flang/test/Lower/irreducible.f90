! RUN: bbc %s -o "-" | FileCheck %s

      ! CHECK-LABEL: irreducible
      subroutine irreducible(k)
        ! CHECK: cond_br %{{[0-9]+}}, ^bb1, ^bb2
        if (k < 5) goto 20
        ! CHECK: ^bb1:  // 2 preds: ^bb0, ^bb2
10      print*, k                             ! scc entry #1: (k < 5) is false
        k = k + 1
        ! CHECK: ^bb2:  // 2 preds: ^bb0, ^bb1
        ! CHECK: cond_br %{{[0-9]+}}, ^bb1, ^bb3
20      if (k < 3) goto 10                    ! scc entry #2: (k < 5) is true
        ! CHECK: ^bb3:  // pred: ^bb2
      end

      ! CHECK-LABEL: main
      program p
        do i = 0, 6
          n = i
          print*
          print*, 1000 + n
          call irreducible(n)
        enddo
      end
