! Check Flang on AIX OBJECT_MODE handling and -maix* flag behavior.
!REQUIRES: system-aix

!RUN: env -u OBJECT_MODE not %flang -print-target-triple 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=32 not %flang -print-target-triple 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=32_64 not %flang -print-target-triple 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=any not %flang -print-target-triple 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=64 %flang -print-target-triple 2>&1 | FileCheck -check-prefix=MODE-64BIT %s

!RUN: not %flang -maix32 -print-target-triple 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=64 not %flang -maix32 -print-target-triple 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=32 %flang -maix64 -print-target-triple 2>&1 | FileCheck -check-prefix=MODE-64BIT %s

!MODE-64BIT: powerpc64-ibm-aix
!MAIX32-ERROR: error: the 32-bit compile mode is not supported. Use OBJECT_MODE=64, -maix64 or -m64
