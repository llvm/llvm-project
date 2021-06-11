! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Simple check that if constructs are ok.

if (a < b) then
  a = 1
end if

if (a < b) then
  a = 2
else
  a = 3
endif

if (a < b) then
  a = 4
else if(a == b) then
  a = 5
end if

if (a < b) then
  a = 6
else if(a == b) then
  a = 7
elseif(a > b) then
  a = 8
end if

if (a < b) then
  a = 9
else if(a == b) then
  a = 10
else
  a = 11
end if

if (a < b) then
  a = 12
else if(a == b) then
  a = 13
else if(a > b) then
  a = 14
end if

if (f()) then
  a = 15
end if

contains
  logical function f()
    f = .true.
  end
end
