! Test handling of -fsplit-machine-functions and -fno-split-machine-functions.

! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s --check-prefix=NO-SPLIT-MACHINE-FUNCTIONS  %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=SPLIT-MACHINE-FUNCTIONS  %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=NO-SPLIT-MACHINE-FUNCTIONS  %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fsplit-machine-functions -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=NO-SPLIT-MACHINE-FUNCTIONS  %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fno-split-machine-functions -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=SPLIT-MACHINE-FUNCTIONS  %}
! RUN: %if arm-registered-target %{ not %flang -### --target=arm-unknown-linux -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-OPT %}
! RUN: %if arm-registered-target %{ %flang -### --target=arm-unknown-linux -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=NO-SPLIT-MACHINE-FUNCTIONS %}

! SPLIT-MACHINE-FUNCTIONS: "-fsplit-machine-functions"
! NO-SPLIT-MACHINE-FUNCTIONS-NOT: "-fsplit-machine-functions"
! UNSUPPORTED-OPT: error: unsupported option '-fsplit-machine-functions' for target

