! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck --allow-empty %s
!CHECK-NOT: error:
 1    function fun()
 2    end function
 3    write(6,*) "pass"
 4    end

