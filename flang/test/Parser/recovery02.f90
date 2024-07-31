! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
continue ! force executable part
CALL ADD_HASH_BLOCK(d_c,f_c,dimc, &
  (h2b-1+noab*(h1b-1+noab*(p4b-noab-1+nvab*(p3b-noab-1$)))))
end

!CHECK: error: expected ')'
!CHECK: in the context: CALL statement
