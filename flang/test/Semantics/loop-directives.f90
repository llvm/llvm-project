! RUN: %python %S/test_errors.py %s %flang

program empty
  ! ERROR: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
end program empty

program non_do
  ! ERROR: A DO loop must follow the VECTOR ALWAYS directive
  !dir$ vector always
  a = 1
end program non_do

