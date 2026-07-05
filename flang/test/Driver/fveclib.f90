! RUN: %flang -### -c -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB %s
! RUN: %flang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck -check-prefix CHECK-ACCELERATE %s
! RUN: %flang -### -c --target=x86_64-unknown-linux-gnu -fveclib=libmvec %s 2>&1 | FileCheck -check-prefix CHECK-libmvec %s
! RUN: %flang -### -c --target=aarch64-unknown-linux-gnu -fveclib=libmvec %s 2>&1 | FileCheck -check-prefix CHECK-libmvec %s
! RUN: %flang -### -c -fveclib=MASSV %s 2>&1 | FileCheck -check-prefix CHECK-MASSV %s
! RUN: %flang -### -c -fveclib=Darwin_libsystem_m %s 2>&1 | FileCheck -check-prefix CHECK-DARWIN_LIBSYSTEM_M %s
! RUN: %flang -### -c --target=aarch64-none-none -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-SLEEF %s
! RUN: %flang -### -c --target=aarch64-none-none -fveclib=ArmPL %s 2>&1 | FileCheck -check-prefix CHECK-ARMPL %s
! RUN: %flang -### -c --target=x86_64-unknown-linux-gnu -fveclib=AMDLIBM %s 2>&1 | FileCheck -check-prefix CHECK-AMDLIBM %s
! RUN: %flang -### -c --target=aarch64-apple-darwin -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB-DARWIN %s
! RUN: not %flang -c -fveclib=something %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

! CHECK-NOLIB: "-fveclib=none"
! CHECK-ACCELERATE: "-fveclib=Accelerate"
! CHECK-libmvec: "-fveclib=libmvec"
! CHECK-MASSV: "-fveclib=MASSV"
! CHECK-DARWIN_LIBSYSTEM_M: "-fveclib=Darwin_libsystem_m"
! CHECK-SLEEF: "-fveclib=SLEEF"
! CHECK-ARMPL: "-fveclib=ArmPL"
! CHECK-AMDLIBM: "-fveclib=AMDLIBM"
! CHECK-NOLIB-DARWIN: "-fveclib=none"

! CHECK-INVALID: error: invalid value 'something' in '-fveclib=something'

! RUN: not %flang --target=x86-none-none -c -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=x86-none-none -c -fveclib=ArmPL %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=riscv64-none-none -c -fveclib=libmvec %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=aarch64-none-none -c -fveclib=SVML %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=aarch64-none-none -c -fveclib=AMDLIBM %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! CHECK-ERROR: unsupported option {{.*}} for target

! RUN: %flang -fveclib=Accelerate %s -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK %s
! CHECK-LINK: "-framework" "Accelerate"

! TODO: if we add support for -nostdlib or -nodefaultlibs we need to test that
! these prevent "-framework Accelerate" being added on Darwin

! RUN: %flang -### --target=aarch64-pc-windows-msvc -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-MSVC %s
! RUN: %flang -### --target=aarch64-linux-gnu -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-LINUX %s
! RUN: %flang -### --target=aarch64-linux-gnu -fveclib=ArmPL -nostdlib %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-NOSTDLIB-LINUX %s
! RUN: %flang -### --target=aarch64-linux-gnu -fveclib=ArmPL %s -lamath 2>&1 | FileCheck --check-prefix=CHECK-LINKING-AMATH-BEFORE-ARMPL-LINUX %s
! RUN: %flang -### --target=arm64-apple-darwin -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-DARWIN %s
! RUN: %flang -### --target=arm64-apple-darwin -fveclib=ArmPL -nostdlib %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-NOSTDLIB-DARWIN %s
! RUN: %flang -### --target=arm64-apple-darwin -fveclib=ArmPL %s -lamath 2>&1 | FileCheck --check-prefix=CHECK-LINKING-AMATH-BEFORE-ARMPL-DARWIN %s
! CHECK-LINKING-ARMPL-LINUX: "--push-state" "--as-needed" "-lm" "-lamath" "-lm" "--pop-state"
! CHECK-LINKING-ARMPL-NOSTDLIB-LINUX: "--push-state" "--as-needed" "-lamath" "--pop-state"
! CHECK-LINKING-ARMPL-NOSTDLIB-LINUX-NOT: "-lm"
! CHECK-LINKING-ARMPL-DARWIN: "-lm" "-lamath" "-lm"
! CHECK-LINKING-ARMPL-NOSTDLIB-DARWIN: "-lamath"
! CHECK-LINKING-ARMPL-NOSTDLIB-DARWIN-NOT: "-lm"
! CHECK-LINKING-ARMPL-MSVC: "--dependent-lib=amath"
! CHECK-LINKING-AMATH-BEFORE-ARMPL-LINUX: "-lamath" {{.*}}"--push-state" "--as-needed" "-lm" "-lamath" "-lm" "--pop-state"
! CHECK-LINKING-AMATH-BEFORE-ARMPL-DARWIN: "-lamath" {{.*}}"-lm" "-lamath" "-lm"

! RUN: %flang -### --target=aarch64-linux-gnu -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir -frtlib-add-rpath -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-RPATH-ARMPL %s
! CHECK-RPATH-ARMPL: "--push-state" "--as-needed" "-lm" "-lamath" "-lm" "--pop-state"
! We need to see "-rpath" at least twice, one for veclib, one for the Fortran runtime
! CHECK-RPATH-ARMPL-SAME: "-rpath"
! CHECK-RPATH-ARMPL-SAME: "-rpath"
