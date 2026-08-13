! Tests that -finit-local= and -finit-local-zero are accepted by the Flang
! driver and forwarded correctly to -fc1.

! --- Valid values: zero, nan, snan, hex byte ---
! RUN: %flang -### -S -finit-local=zero  %s -o - 2>&1 | FileCheck --check-prefix=CHECK-ZERO  %s
! RUN: %flang -### -S -finit-local=nan   %s -o - 2>&1 | FileCheck --check-prefix=CHECK-NAN   %s
! RUN: %flang -### -S -finit-local=snan  %s -o - 2>&1 | FileCheck --check-prefix=CHECK-SNAN  %s
! RUN: %flang -### -S -finit-local=0xAA  %s -o - 2>&1 | FileCheck --check-prefix=CHECK-HEX   %s
! RUN: %flang -### -S -finit-local=0xff  %s -o - 2>&1 | FileCheck --check-prefix=CHECK-HEX2  %s

! --- GFortran alias: -finit-local-zero ---
! RUN: %flang -### -S -finit-local-zero  %s -o - 2>&1 | FileCheck --check-prefix=CHECK-ALIAS %s

! --- Compiler (fc1) directly accepts -finit-local= ---
! RUN: %flang_fc1 -emit-hlfir -finit-local=zero  %s -o -
! RUN: %flang_fc1 -emit-hlfir -finit-local=nan   %s -o -
! RUN: %flang_fc1 -emit-hlfir -finit-local=snan  %s -o -
! RUN: %flang_fc1 -emit-hlfir -finit-local=0xAA  %s -o -
! RUN: %flang_fc1 -emit-hlfir -finit-local-zero  %s -o -

! --- Invalid value should produce a diagnostic (fc1 level) ---
! RUN: not %flang_fc1 -emit-hlfir -finit-local=bogus %s -o - 2>&1 | FileCheck --check-prefix=CHECK-ERR %s

! CHECK-ZERO:  "-fc1"{{.*}}"-finit-local=zero"
! CHECK-NAN:   "-fc1"{{.*}}"-finit-local=nan"
! CHECK-SNAN:  "-fc1"{{.*}}"-finit-local=snan"
! CHECK-HEX:   "-fc1"{{.*}}"-finit-local=0xAA"
! CHECK-HEX2:  "-fc1"{{.*}}"-finit-local=0xff"
! CHECK-ALIAS: "-fc1"{{.*}}"-finit-local=zero"
! CHECK-ERR:   error: invalid value 'bogus' in '-finit-local=bogus'

subroutine dummy_sub()
end subroutine
