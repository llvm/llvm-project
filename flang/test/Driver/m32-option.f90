! Check support of -m32.
! RUN: not %flang -target powerpc-ibm-aix -m32 -### - %s 2>&1 | FileCheck -check-prefix=M32-AIX-ERROR %s
! RUN: not %flang -target powerpc64-ibm-aix -m32 -### - %s 2>&1 | FileCheck -check-prefix=M32-AIX-ERROR %s
! RUN: not %flang -target powerpc-ibm-aix -maix32 -### - %s 2>&1 | FileCheck -check-prefix=M32-AIX-ERROR %s
! RUN: not %flang -target powerpc64-ibm-aix -maix32 -### - %s 2>&1 | FileCheck -check-prefix=M32-AIX-ERROR %s
! RUN: %flang -target powerpc-ibm-aix -maix64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s
! RUN: %flang -target powerpc64-ibm-aix -maix64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s
! RUN: not %flang -target powerpc64le-unknown-linux-gnu -m32 -### - %s 2>&1 | FileCheck -check-prefix=M32-ERROR %s
! RUN: not %flang -target powerpc64le-unknown-linux-gnu -maix32 -### - %s 2>&1 | FileCheck -check-prefix=MAIX32-ERROR %s

! M32-AIX-ERROR: error: the 32-bit compile mode is not supported. Use OBJECT_MODE=64, -maix64 or -m64
! M64: "-triple" "powerpc64-ibm-aix{{.*}}"
! M32-ERROR: error: unsupported option '-m32' for target 'powerpc64le-unknown-linux-gnu'
! MAIX32-ERROR: error: unsupported option '-maix32' for target 'powerpc64le-unknown-linux-gnu'
