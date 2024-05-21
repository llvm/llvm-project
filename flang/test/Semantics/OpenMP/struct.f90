! UNSUPPORTED: system-windows
! Marking as unsupported due to suspected long runtime on Windows
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! Check OpenMP compatibility with the DEC STRUCTURE extension

structure /s/
end structure

end
