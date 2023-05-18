! RUN: %python %S/test_symbols.py %s %flang_fc1
 !DEF: /MainProgram1/pptr EXTERNAL, POINTER ProcEntity
 procedure(), pointer :: pptr
 !REF: /MainProgram1/pptr
 !DEF: /mustbeexternal EXTERNAL ProcEntity
 pptr => mustbeexternal
end program
