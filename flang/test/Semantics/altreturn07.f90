! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s
! Test extension: RETURN from main program

return
! CHECK: portability: RETURN should not appear in a main program
end
