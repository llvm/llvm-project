! RUN: env FCC_OVERRIDE_OPTIONS="#+-Os +-Oz +-O +-O3 +-Oignore +a +b +c xb Xa Omagic ^-###  " %flang --target=x86_64-unknown-linux-gnu %s -O2 b -O3 2>&1 | FileCheck %s
! RUN: env FCC_OVERRIDE_OPTIONS="x-Werror +-g" %flang --target=x86_64-unknown-linux-gnu -Werror %s -c -### 2>&1 | FileCheck %s -check-prefix=RM-WERROR
! RUN: env FCC_OVERRIDE_OPTIONS="x-Werror" %flang --config=%S/Inputs/config-7.cfg -### %s -c  2>&1 | FileCheck %s -check-prefix=CONF

! CHECK: "-fc1"
! CHECK-NOT: "-Oignore"
! CHECK: "-Omagic"
! CHECK-NOT: "-Oignore"

! RM-WERROR: ### FCC_OVERRIDE_OPTIONS: x-Werror +-g
! RM-WERROR-NEXT: ### Deleting argument -Werror
! RM-WERROR-NEXT: ### Adding argument -g at end
! RM-WERROR-NOT: "-Werror"

! Test that FCC_OVERRIDE_OPTIONS does not affect the options from config files.
! CONF: ### FCC_OVERRIDE_OPTIONS: x-Werror
! CONF: "-Werror"
