! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: print *, "HELLO "//" WORLD"
print *, "HELLO "/&
                       &/" WORLD"
end
