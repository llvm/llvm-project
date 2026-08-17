! Tests that -finit-local= and -finit-local-zero are accepted by the Flang
! driver and forwarded correctly to -fc1.

! --- Valid values: zero, hex byte ---
! RUN: %flang -### -S -finit-local=zero  %s 2>&1 | FileCheck --check-prefix=ZERO  %s
! RUN: %flang -### -S -finit-local=0xAA  %s 2>&1 | FileCheck --check-prefix=HEX   %s
! RUN: %flang -### -S -finit-local=0xff  %s 2>&1 | FileCheck --check-prefix=HEX2  %s

! --- GFortran alias: -finit-local-zero ---
! RUN: %flang -### -S -finit-local-zero  %s 2>&1 | FileCheck --check-prefix=ZERO %s
! --- Invalid value should produce a diagnostic (fc1 level) ---
! RUN: not %flang_fc1 -emit-hlfir -finit-local=bogus %s 2>&1 | FileCheck --check-prefix=ERR %s

! ZERO:  "-fc1"{{.*}} "-finit-local=zero"
! HEX:   "-fc1"{{.*}} "-finit-local=0xAA"
! HEX2:  "-fc1"{{.*}} "-finit-local=0xff"
! ERR:   error: invalid value 'bogus' in '-finit-local=bogus'

subroutine dummy_sub()
end subroutine
