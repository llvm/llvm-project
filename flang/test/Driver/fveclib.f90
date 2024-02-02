! RUN: %flang -### -c -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB %s
! RUN: %flang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck -check-prefix CHECK-ACCELERATE %s
! RUN: %flang -### -c -fveclib=libmvec %s 2>&1 | FileCheck -check-prefix CHECK-libmvec %s
! RUN: %flang -### -c -fveclib=MASSV %s 2>&1 | FileCheck -check-prefix CHECK-MASSV %s
! RUN: %flang -### -c -fveclib=Darwin_libsystem_m %s 2>&1 | FileCheck -check-prefix CHECK-DARWIN_LIBSYSTEM_M %s
! RUN: %flang -### -c --target=aarch64-none-none -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-SLEEF %s
! RUN: %flang -### -c --target=aarch64-none-none -fveclib=ArmPL %s 2>&1 | FileCheck -check-prefix CHECK-ARMPL %s
! RUN: %flang -### -c --target=aarch64-apple-darwin -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB-DARWIN %s
! RUN: not %flang -c -fveclib=something %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

! CHECK-NOLIB: "-fveclib=none"
! CHECK-ACCELERATE: "-fveclib=Accelerate"
! CHECK-libmvec: "-fveclib=libmvec"
! CHECK-MASSV: "-fveclib=MASSV"
! CHECK-DARWIN_LIBSYSTEM_M: "-fveclib=Darwin_libsystem_m"
! CHECK-SLEEF: "-fveclib=SLEEF"
! CHECK-ARMPL: "-fveclib=ArmPL"
! CHECK-NOLIB-DARWIN: "-fveclib=none"

! CHECK-INVALID: error: invalid value 'something' in '-fveclib=something'

! RUN: not %flang --target=x86-none-none -c -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=x86-none-none -c -fveclib=ArmPL %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=aarch64-none-none -c -fveclib=LIBMVEC-X86 %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=aarch64-none-none -c -fveclib=SVML %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! CHECK-ERROR: unsupported option {{.*}} for target

! RUN: %flang -fveclib=Accelerate %s -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK %s
! CHECK-LINK: "-framework" "Accelerate"

! TODO: if we add support for -nostdlib or -nodefaultlibs we need to test that
! these prevent "-framework Accelerate" being added on Darwin

! Verify that the correct vector library is passed to LTO flags.

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fveclib=LIBMVEC -flto %s 2>&1 | FileCheck -check-prefix CHECK-LTO-LIBMVEC %s
! CHECK-LTO-LIBMVEC: "-plugin-opt=-vector-library=LIBMVEC-X86"

! RUN: %flang -### --target=powerpc64-unknown-linux-gnu -fveclib=MASSV -flto %s 2>&1 | FileCheck -check-prefix CHECK-LTO-MASSV %s
! CHECK-LTO-MASSV: "-plugin-opt=-vector-library=MASSV"

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fveclib=SVML -flto %s 2>&1 | FileCheck -check-prefix CHECK-LTO-SVML %s
! CHECK-LTO-SVML: "-plugin-opt=-vector-library=SVML"

! RUN: %flang -### --target=aarch64-linux-gnu -fveclib=SLEEF -flto %s 2>&1 | FileCheck -check-prefix CHECK-LTO-SLEEF %s
! CHECK-LTO-SLEEF: "-plugin-opt=-vector-library=sleefgnuabi"

! RUN: %flang -### --target=aarch64-linux-gnu -fveclib=ArmPL -flto %s 2>&1 | FileCheck -check-prefix CHECK-LTO-ARMPL %s
! CHECK-LTO-ARMPL: "-plugin-opt=-vector-library=ArmPL"
