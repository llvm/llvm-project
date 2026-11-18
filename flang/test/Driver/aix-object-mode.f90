! Check Flang on AIX OBJECT_MODE handling with -maix* setting.
!REQUIRES: system-aix

!RUN: env -u OBJECT_MODE not %flang %s 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=32 not %flang %s 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=32_64 not %flang %s 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=any not %flang %s 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=64 %flang -print-target-triple %s 2>&1 | FileCheck -check-prefix=MODE-64BIT %s

!RUN: env OBJECT_MODE=64 not %flang -maix32 %s 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

!RUN: env OBJECT_MODE=32 %flang -maix64 -print-target-triple %s 2>&1 | FileCheck -check-prefix=MODE-64BIT %s

!RUN: env OBJECT_MODE=7 not %flang %s 2>&1 | FileCheck -check-prefix=OBJECT-MODE-INVALID-ERROR %s

!RUN: env OBJECT_MODE='' not %flang %s 2>&1 | FileCheck -check-prefix=OBJECT-MODE-EMPTY-ERROR %s

!OBJECT-MODE-INVALID-ERROR: error: OBJECT_MODE setting 7 is not recognized and is not a valid setting
!OBJECT-MODE-EMPTY-ERROR: error: OBJECT_MODE setting is not recognized and is not a valid setting
!MODE-64BIT: powerpc64-ibm-aix
!MAIX32-ERROR: error: 32-bit compile mode is not supported. Use OBJECT_MODE=64, -maix64 or -m64

program main
end
