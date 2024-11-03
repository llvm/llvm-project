// RUN: %clang -### -c -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB %s
// RUN: %clang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck -check-prefix CHECK-ACCELERATE %s
// RUN: %clang -### -c -fveclib=libmvec %s 2>&1 | FileCheck -check-prefix CHECK-libmvec %s
// RUN: %clang -### -c -fveclib=MASSV %s 2>&1 | FileCheck -check-prefix CHECK-MASSV %s
// RUN: %clang -### -c -fveclib=Darwin_libsystem_m %s 2>&1 | FileCheck -check-prefix CHECK-DARWIN_LIBSYSTEM_M %s
// RUN: %clang -### -c --target=aarch64-none-none -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-SLEEF %s
// RUN: not %clang -c -fveclib=something %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

// CHECK-NOLIB: "-fveclib=none"
// CHECK-ACCELERATE: "-fveclib=Accelerate"
// CHECK-libmvec: "-fveclib=libmvec"
// CHECK-MASSV: "-fveclib=MASSV"
// CHECK-DARWIN_LIBSYSTEM_M: "-fveclib=Darwin_libsystem_m"
// CHECK-SLEEF: "-fveclib=SLEEF"

// CHECK-INVALID: error: invalid value 'something' in '-fveclib=something'

// RUN: not %clang --target=x86-none-none -c -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
// RUN: not %clang --target=aarch64-none-none -c -fveclib=LIBMVEC-X86 %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
// RUN: not %clang --target=aarch64-none-none -c -fveclib=SVML %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
// CHECK-ERROR: unsupported option {{.*}} for target

// RUN: %clang -fveclib=Accelerate %s -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK %s
// CHECK-LINK: "-framework" "Accelerate"

// RUN: %clang -fveclib=Accelerate %s -nostdlib -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK-NOSTDLIB %s
// CHECK-LINK-NOSTDLIB-NOT: "-framework" "Accelerate"

// RUN: %clang -fveclib=Accelerate %s -nodefaultlibs -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK-NODEFAULTLIBS %s
// CHECK-LINK-NODEFAULTLIBS-NOT: "-framework" "Accelerate"
