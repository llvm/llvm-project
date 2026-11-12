!RUN: %flang -c %s -### 2>&1
function s(x) result(i)
!CHECK-WARNING: Function result is never defined
integer::x
procedure():: i
end function
end
