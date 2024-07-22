// RUN: %clang -### -c -fveclib=none %s 2>&1 | FileCheck --check-prefix=CHECK-NOLIB %s
// RUN: %clang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck --check-prefix=CHECK-ACCELERATE %s
// RUN: %clang -### -c -fveclib=libmvec %s 2>&1 | FileCheck --check-prefix=CHECK-libmvec %s
// RUN: %clang -### -c -fveclib=MASSV %s 2>&1 | FileCheck --check-prefix=CHECK-MASSV %s
// RUN: %clang -### -c -fveclib=Darwin_libsystem_m %s 2>&1 | FileCheck --check-prefix=CHECK-DARWIN_LIBSYSTEM_M %s
// RUN: %clang -### -c --target=aarch64 -fveclib=SLEEF %s 2>&1 | FileCheck --check-prefix=CHECK-SLEEF %s
// RUN: %clang -### -c --target=aarch64 -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-ARMPL %s
// RUN: not %clang -c -fveclib=something %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s

// CHECK-NOLIB: "-fveclib=none"
// CHECK-ACCELERATE: "-fveclib=Accelerate"
// CHECK-libmvec: "-fveclib=libmvec"
// CHECK-MASSV: "-fveclib=MASSV"
// CHECK-DARWIN_LIBSYSTEM_M: "-fveclib=Darwin_libsystem_m"
// CHECK-SLEEF: "-fveclib=SLEEF"
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

// RUN: %clang -### --target=aarch64-linux-gnu -fveclib=ArmPL -flto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-ARMPL %s
// CHECK-LTO-ARMPL: "-plugin-opt=-vector-library=ArmPL"
