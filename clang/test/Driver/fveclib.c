// RUN: %clang -### -c -fveclib=none %s 2>&1 | FileCheck --check-prefix=CHECK-NOLIB %s
// RUN: %clang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck --check-prefix=CHECK-ACCELERATE %s
// RUN: %clang -### -c -fveclib=libmvec %s 2>&1 | FileCheck --check-prefix=CHECK-libmvec %s
// RUN: %clang -### -c -fveclib=MASSV %s 2>&1 | FileCheck --check-prefix=CHECK-MASSV %s
// RUN: %clang -### -c -fveclib=Darwin_libsystem_m %s 2>&1 | FileCheck --check-prefix=CHECK-DARWIN_LIBSYSTEM_M %s
// RUN: %clang -### -c --target=aarch64 -fveclib=SLEEF %s 2>&1 | FileCheck --check-prefix=CHECK-SLEEF %s
// RUN: %clang -### -c --target=riscv64-unknown-linux-gnu -fveclib=SLEEF -march=rv64gcv %s 2>&1 | FileCheck -check-prefix CHECK-SLEEF-RISCV %s
// RUN: %clang -### -c --target=aarch64 -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-ARMPL %s
// RUN: not %clang -c -fveclib=something %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s

// CHECK-NOLIB: "-fveclib=none"
// CHECK-ACCELERATE: "-fveclib=Accelerate"
// CHECK-libmvec: "-fveclib=libmvec"
// CHECK-MASSV: "-fveclib=MASSV"
// CHECK-DARWIN_LIBSYSTEM_M: "-fveclib=Darwin_libsystem_m"
// CHECK-SLEEF: "-fveclib=SLEEF"
// CHECK-SLEEF-RISCV: "-fveclib=SLEEF"
// CHECK-ARMPL: "-fveclib=ArmPL"

// CHECK-INVALID: error: invalid value 'something' in '-fveclib=something'

// RUN: not %clang --target=x86 -c -fveclib=SLEEF %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not %clang --target=x86 -c -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not %clang --target=aarch64 -c -fveclib=LIBMVEC-X86 %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not %clang --target=aarch64 -c -fveclib=SVML %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// CHECK-ERROR: unsupported option {{.*}} for target

// RUN: %clang -fveclib=Accelerate %s -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK %s
// CHECK-LINK: "-framework" "Accelerate"

// RUN: %clang -fveclib=Accelerate %s -nostdlib -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK-NOSTDLIB %s
// CHECK-LINK-NOSTDLIB-NOT: "-framework" "Accelerate"

// RUN: %clang -fveclib=Accelerate %s -nodefaultlibs -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK-NODEFAULTLIBS %s
// CHECK-LINK-NODEFAULTLIBS-NOT: "-framework" "Accelerate"


/* Verify that the correct vector library is passed to LTO flags. */

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fveclib=LIBMVEC -flto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-LIBMVEC %s
// CHECK-LTO-LIBMVEC: "-plugin-opt=-vector-library=LIBMVEC-X86"

// RUN: %clang -### --target=powerpc64-unknown-linux-gnu -fveclib=MASSV -flto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-MASSV %s
// CHECK-LTO-MASSV: "-plugin-opt=-vector-library=MASSV"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fveclib=SVML -flto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-SVML %s
// CHECK-LTO-SVML: "-plugin-opt=-vector-library=SVML"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=SLEEF -flto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-SLEEF %s
// CHECK-LTO-SLEEF: "-plugin-opt=-vector-library=sleefgnuabi"

// RUN: %clang -### --target=riscv64-unknown-linux-gnu -fveclib=SLEEF -flto -march=rv64gcv %s 2>&1 | FileCheck -check-prefix CHECK-LTO-SLEEF-RISCV %s
// CHECK-LTO-SLEEF-RISCV: "-plugin-opt=-vector-library=sleefgnuabi"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -flto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-ARMPL %s
// CHECK-LTO-ARMPL: "-plugin-opt=-vector-library=ArmPL"


/* Verify that -fmath-errno is set correctly for the vector library. */

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fveclib=LIBMVEC %s 2>&1 | FileCheck --check-prefix=CHECK-ERRNO-LIBMVEC %s
// CHECK-ERRNO-LIBMVEC: "-fveclib=LIBMVEC"
// CHECK-ERRNO-LIBMVEC-SAME: "-fmath-errno"

// RUN: %clang -### --target=powerpc64-unknown-linux-gnu -fveclib=MASSV %s 2>&1 | FileCheck --check-prefix=CHECK-ERRNO-MASSV %s
// CHECK-ERRNO-MASSV: "-fveclib=MASSV"
// CHECK-ERRNO-MASSV-SAME: "-fmath-errno"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fveclib=SVML %s 2>&1 | FileCheck --check-prefix=CHECK-ERRNO-SVML %s
// CHECK-ERRNO-SVML: "-fveclib=SVML"
// CHECK-ERRNO-SVML-SAME: "-fmath-errno"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=SLEEF %s 2>&1 | FileCheck --check-prefix=CHECK-ERRNO-SLEEF %s
// CHECK-ERRNO-SLEEF-NOT: "-fmath-errno"
// CHECK-ERRNO-SLEEF: "-fveclib=SLEEF"
// CHECK-ERRNO-SLEEF-NOT: "-fmath-errno"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-ERRNO-ARMPL %s
// CHECK-ERRNO-ARMPL-NOT: "-fmath-errno"
// CHECK-ERRNO-ARMPL: "-fveclib=ArmPL"
// CHECK-ERRNO-ARMPL-NOT: "-fmath-errno"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -fmath-errno %s 2>&1 | FileCheck --check-prefix=CHECK-REENABLE-ERRNO-ARMPL %s
// CHECK-REENABLE-ERRNO-ARMPL: math errno enabled by '-fmath-errno' after it was implicitly disabled by '-fveclib=ArmPL', this may limit the utilization of the vector library [-Wmath-errno-enabled-with-veclib]
// CHECK-REENABLE-ERRNO-ARMPL: "-fveclib=ArmPL"
// CHECK-REENABLE-ERRNO-ARMPL-SAME: "-fmath-errno"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=SLEEF -fmath-errno %s 2>&1 | FileCheck --check-prefix=CHECK-REENABLE-ERRNO-SLEEF %s
// CHECK-REENABLE-ERRNO-SLEEF: math errno enabled by '-fmath-errno' after it was implicitly disabled by '-fveclib=SLEEF', this may limit the utilization of the vector library [-Wmath-errno-enabled-with-veclib]
// CHECK-REENABLE-ERRNO-SLEEF: "-fveclib=SLEEF"
// CHECK-REENABLE-ERRNO-SLEEF-SAME: "-fmath-errno"

// RUN: %clang -### --target=riscv64-unknown-linux-gnu -fveclib=SLEEF -fmath-errno -march=rv64gcv %s 2>&1 | FileCheck --check-prefix=CHECK-REENABLE-ERRNO-SLEEF-RISCV %s
// CHECK-REENABLE-ERRNO-SLEEF-RISCV: math errno enabled by '-fmath-errno' after it was implicitly disabled by '-fveclib=SLEEF', this may limit the utilization of the vector library [-Wmath-errno-enabled-with-veclib]
// CHECK-REENABLE-ERRNO-SLEEF-RISCV: "-fveclib=SLEEF"
// CHECK-REENABLE-ERRNO-SLEEF-RISCV-SAME: "-fmath-errno"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -fno-fast-math %s 2>&1 | FileCheck --check-prefix=CHECK-REENABLE-ERRNO-NFM %s
// CHECK-REENABLE-ERRNO-NFM: math errno enabled by '-fno-fast-math' after it was implicitly disabled by '-fveclib=ArmPL', this may limit the utilization of the vector library [-Wmath-errno-enabled-with-veclib]
// CHECK-REENABLE-ERRNO-NFM: "-fveclib=ArmPL"
// CHECK-REENABLE-ERRNO-NFM-SAME: "-fmath-errno"

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -ffp-model=strict %s 2>&1 | FileCheck --check-prefix=CHECK-REENABLE-ERRNO-FP-MODEL %s
// CHECK-REENABLE-ERRNO-FP-MODEL: math errno enabled by '-ffp-model=strict' after it was implicitly disabled by '-fveclib=ArmPL', this may limit the utilization of the vector library [-Wmath-errno-enabled-with-veclib]
// CHECK-REENABLE-ERRNO-FP-MODEL: "-fveclib=ArmPL"
// CHECK-REENABLE-ERRNO-FP-MODEL-SAME: "-fmath-errno"

/* Verify the warning points at the last arg to enable -fmath-errno. */
// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -fno-fast-math -fno-math-errno -ffp-model=strict %s 2>&1 | FileCheck --check-prefix=CHECK-ENABLED-LAST %s
// CHECK-ENABLED-LAST: math errno enabled by '-ffp-model=strict' after it was implicitly disabled by '-fveclib=ArmPL', this may limit the utilization of the vector library [-Wmath-errno-enabled-with-veclib]

/* Verify no warning when math-errno is re-enabled for a different veclib (that does not imply -fno-math-errno). */
// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -fmath-errno -fveclib=LIBMVEC %s 2>&1 | FileCheck --check-prefix=CHECK-REPEAT-VECLIB %s
// CHECK-REPEAT-VECLIB-NOT: math errno enabled

/// Verify that vectorized routines library is being linked in.
// RUN: %clang -### --target=aarch64-pc-windows-msvc -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-MSVC %s
// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-LINUX %s
// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL %s -lamath 2>&1 | FileCheck --check-prefix=CHECK-LINKING-AMATH-BEFORE-ARMPL-LINUX %s
// RUN: %clang -### --target=arm64-apple-darwin -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-DARWIN %s
// RUN: %clang -### --target=arm64-apple-darwin -fveclib=ArmPL %s -lamath 2>&1 | FileCheck --check-prefix=CHECK-LINKING-AMATH-BEFORE-ARMPL-DARWIN %s
// CHECK-LINKING-ARMPL-LINUX: "--push-state" "--as-needed" "-lm" "-lamath" "-lm" "--pop-state"
// CHECK-LINKING-ARMPL-DARWIN: "-lm" "-lamath" "-lm"
// CHECK-LINKING-ARMPL-MSVC: "--dependent-lib=amath"
// CHECK-LINKING-AMATH-BEFORE-ARMPL-LINUX: "-lamath" {{.*}}"--push-state" "--as-needed" "-lm" "-lamath" "-lm" "--pop-state"
// CHECK-LINKING-AMATH-BEFORE-ARMPL-DARWIN: "-lamath" {{.*}}"-lm" "-lamath" "-lm"

/// Verify that the RPATH is being set when needed.
// RUN: %clang -### --target=aarch64-linux-gnu -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir -frtlib-add-rpath -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-RPATH-ARMPL %s
// CHECK-RPATH-ARMPL: "--push-state" "--as-needed" "-lm" "-lamath" "-lm" "--pop-state"
// CHECK-RPATH-ARMPL-SAME: "-rpath"
