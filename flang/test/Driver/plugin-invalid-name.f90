! Check the correct error diagnostic is reported when a plugin name isn't found

! REQUIRES: plugins, shell

! RUN: not %flang_fc1 -plugin -wrong-name %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: error: unable to find plugin '-wrong-name'
