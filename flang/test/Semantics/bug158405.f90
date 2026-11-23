! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s()
  ! ERROR: Generic 'f' may not have specific procedures 's' and 'ss' as their interfaces are not distinguishable
  interface f
    procedure s
    procedure ss
  end interface
 entry ss()
end
