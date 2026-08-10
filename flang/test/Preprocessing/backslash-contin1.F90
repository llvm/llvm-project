! RUN: %flang -E %s | FileCheck %s
print *, \
  "hello, \
world"
end
!CHECK:      print *, "hello, world"
!CHECK:      end

