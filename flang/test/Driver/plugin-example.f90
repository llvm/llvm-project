! Check that loading and running the Hello World plugin example results in the correct print statement
! Also check that when a plugin name isn't found, the error diagnostic is correct
! This requires that the examples are built (FLANG_BUILD_EXAMPLES=ON)

! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangHelloWorldPlugin%pluginext -plugin -hello-world %s 2>&1 | FileCheck %s
! CHECK: Hello World from your new Flang plugin

! RUN: not %flang_fc1 -load %llvmshlibdir/flangHelloWorldPlugin%pluginext -plugin -wrong-name %s 2>&1 | FileCheck %s --check-prefix=ERROR
! ERROR: error: unable to find plugin '-wrong-name'
