!RUN: %python %S/test_errors.py %s %flang_fc1 -J %S/Inputs
!ERROR: Cannot read module file for module 'modfile83': 'modfile83.mod' is not a module file for this compiler
use modfile83
end
