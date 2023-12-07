// Make sure that the Driver passes down -stdlib=libc++ to CC1 when specified explicitly.
//
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch arm64  -miphoneos-version-min=7.0 -stdlib=libc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/              -mmacosx-version-min=10.9  -stdlib=libc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=7.0 -stdlib=libc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7k                            -stdlib=libc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBCXX %s
// CHECK-LIBCXX: "-stdlib=libc++"
// CHECK-LIBCXX-NOT: "-stdlib=libstdc++"

// Make sure that the Driver passes down -stdlib=libstdc++ to CC1 when specified explicitly. Note that this
// shouldn't really be used on Darwin because libstdc++ is not supported anymore, but we still pin down the
// Driver behavior for now.
//
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch arm64  -miphoneos-version-min=7.0 -stdlib=libstdc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBSTDCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/              -mmacosx-version-min=10.9  -stdlib=libstdc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBSTDCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=7.0 -stdlib=libstdc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBSTDCXX %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7k                            -stdlib=libstdc++ %s -### 2>&1 | FileCheck --check-prefix=CHECK-LIBSTDCXX %s
// CHECK-LIBSTDCXX: "-stdlib=libstdc++"
// CHECK-LIBSTDCXX-NOT: "-stdlib=libc++"

// Make sure that the Driver does not spuriously pass down any -stdlib=<...> option to CC1 if none is
// specified on the command-line. In that case, CC1 should use the default standard library, which is
// going to be libc++.
//
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch arm64  -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/              -mmacosx-version-min=10.9  %s -### 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7s -miphoneos-version-min=7.0 %s -### 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -target x86_64-apple-darwin -ccc-install-dir %S/Inputs/darwin_toolchain_tree/bin/ -arch armv7k                            %s -### 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// CHECK-NONE-NOT: "-stdlib="
