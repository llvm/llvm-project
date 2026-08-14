! Tests that -finit-local= and -finit-local-zero are accepted by the Flang
! driver and forwarded correctly to -fc1.

! --- Valid values: zero, nan, snan, hex byte ---
! RUN: %flang -### -S -finit-local=zero  %s 2>&1 | FileCheck --check-prefix=ZERO  %s
! RUN: %flang -### -S -finit-local=nan   %s 2>&1 | FileCheck --check-prefix=NAN   %s
! RUN: %flang -### -S -finit-local=snan  %s 2>&1 | FileCheck --check-prefix=SNAN  %s
! RUN: %flang -### -S -finit-local=0xAA  %s 2>&1 | FileCheck --check-prefix=HEX   %s
! RUN: %flang -### -S -finit-local=0xff  %s 2>&1 | FileCheck --check-prefix=HEX2  %s

! --- GFortran alias: -finit-local-zero ---
! RUN: %flang -### -S -finit-local-zero  %s 2>&1 | FileCheck --check-prefix=ZERO %s
! --- Invalid value should produce a diagnostic (fc1 level) ---
! RUN: not %flang_fc1 -emit-hlfir -finit-local=bogus %s 2>&1 | FileCheck --check-prefix=ERR %s

! ZERO:  "-fc1"{{.*}} "-finit-local=zero"
! NAN:   "-fc1"{{.*}} "-finit-local=nan"
! SNAN:  "-fc1"{{.*}} "-finit-local=snan"
! HEX:   "-fc1"{{.*}} "-finit-local=0xAA"
! HEX2:  "-fc1"{{.*}} "-finit-local=0xff"
! ERR:   error: invalid value 'bogus' in '-finit-local=bogus'

subroutine dummy_sub()
end subroutine
