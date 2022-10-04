! Check the correct error diagnostic is reported when a pass plugin shared object isn't found

! REQUIRES: plugins, shell

! RUN: not %flang -fpass-plugin=X.Y %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang_fc1 -emit-llvm -o /dev/null -fpass-plugin=X.Y %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: error: unable to load plugin 'X.Y': 'Could not load library 'X.Y': X.Y: cannot open shared object file: No such file or directory'
