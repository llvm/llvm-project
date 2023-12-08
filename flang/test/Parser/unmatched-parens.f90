! RUN: not %flang_fc1 -E %s 2>&1 | FileCheck %s
do i = 1,10
  ! CHECK: Unmatched '('
  if (i != 0) then
    exit
  endif
enddo
end
