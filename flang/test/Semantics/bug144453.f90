!RUN: %python %S/test_errors.py %s %flang_fc1
function func(immutable)
  logical func
  logical, intent(in) :: immutable
  !No warning about an undefined function result should appear
  INQUIRE(file="/usr/local/games/adventure", EXIST=func)
  !ERROR: EXIST variable 'immutable' is not definable
  !BECAUSE: 'immutable' is an INTENT(IN) dummy argument
  INQUIRE(file="/usr/local/games/adventure", EXIST=immutable)
end
