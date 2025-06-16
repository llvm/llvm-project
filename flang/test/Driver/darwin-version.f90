! Based on clang's darwin-version.c test with tests for ios watchos and tvos
! removed

! RUN: %flang -target i686-apple-darwin8 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
! RUN: %flang -target i686-apple-darwin9 -mmacos-version-min=10.4 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
! CHECK-VERSION-OSX4: "i686-apple-macosx10.4.0"
! RUN: %flang -target i686-apple-darwin9 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
! RUN: %flang -target i686-apple-darwin9 -mmacos-version-min=10.5 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
! CHECK-VERSION-OSX5: "i686-apple-macosx10.5.0"
! RUN: %flang -target i686-apple-darwin10 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
! RUN: %flang -target i686-apple-darwin9 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
! CHECK-VERSION-OSX6: "i686-apple-macosx10.6.0"
! RUN: %flang -target x86_64-apple-darwin14 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
! RUN: %flang -target x86_64-apple-darwin -mmacos-version-min=10.10 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
! RUN: %flang -target x86_64-apple-darwin -mmacos-version-min=10.10 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
! CHECK-VERSION-OSX10: "x86_64-apple-macosx10.10.0"
! RUN: not %flang -target x86_64-apple-darwin -mmacos-version-min= -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-MISSING %s
! RUN: not %flang -target x86_64-apple-darwin -mmacos-version-min= -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-MISSING %s
! CHECK-VERSION-MISSING: missing version number

! RUN: %flang -target x86_64-apple-driverkit19.0 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-DRIVERKIT190 %s
! CHECK-VERSION-DRIVERKIT190: "x86_64-apple-driverkit19.0.0"

! Check environment variable gets interpreted correctly
! RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 IPHONEOS_DEPLOYMENT_TARGET=2.0 \
! RUN:   %flang -target i686-apple-darwin9 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s

! RUN: env MACOSX_DEPLOYMENT_TARGET=10.4.10 \
! RUN:   %flang -target i386-apple-darwin9 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX49 %s
! CHECK-VERSION-OSX49: "i386-apple-macosx10.4.10"
! RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.3.1 \

! Target can specify the OS version:

! RUN: %flang -target x86_64-apple-macos10.11.2 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-TMAC2 %s
! CHECK-VERSION-TMAC2: "x86_64-apple-macosx10.11.2"

! Warn about -m<os>-version-min when it's used with target:

! RUN: %flang -target x86_64-apple-macos10.11.2 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-OSV1 %s
! CHECK-VERSION-TNO-OSV1: overriding '-mmacos-version-min=10.6' option with '-target x86_64-apple-macos10.11.2'

! RUN: %flang -target x86_64-apple-macos10.6 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-TNO-SAME %s
! CHECK-VERSION-TNO-SAME-NOT: overriding
! CHECK-VERSION-TNO-SAME-NOT: argument unused during compilation

! Target with OS version is not overridden by -m<os>-version-min variables:

! RUN: %flang -target x86_64-apple-macos10.11.2 -mmacos-version-min=10.6 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-OSV1 %s
! CHECK-VERSION-TIGNORE-OSV1: "x86_64-apple-macosx10.11.2"

! Target without OS version includes the OS given by -m<os>-version-min arguments:

! RUN: %flang -target x86_64-apple-macos -mmacos-version-min=10.11 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-USE-OS-ARG1 %s
! CHECK-VERSION-USE-OS-ARG1: "x86_64-apple-macosx10.11.0"

! Target with OS version is not overridden by environment variables:

! RUN: env MACOSX_DEPLOYMENT_TARGET=10.1 \
! RUN:   %flang -target i386-apple-macos10.5 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-TMACOS-CMD %s
! CHECK-VERSION-TMACOS-CMD: "i386-apple-macosx10.5.0"

! Target with OS version is not overridden by arch:

! RUN: %flang -target uknown-apple-macos10.11.2 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-VERSION-TIGNORE-ARCH1 %s
! CHECK-VERSION-TIGNORE-ARCH1: "unknown-apple-macosx10.11.2"

! Target can be used to specify the environment:

! RUN: %flang -target x86_64-apple-macos11 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-MACOS11 %s
! RUN: %flang -target x86_64-apple-darwin20 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-MACOS11 %s
! RUN: %flang -target x86_64-apple-darwin -mmacos-version-min=11 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-MACOS11 %s
! CHECK-MACOS11: "x86_64-apple-macosx11.0.0"

! RUN: %flang -target arm64-apple-macosx10.16 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-IMPLICIT-MACOS11 %s
! CHECK-IMPLICIT-MACOS11: warning: overriding deployment version
! CHECK-IMPLICIT-MACOS11: "arm64-apple-macosx11.0.0"

! RUN: %flang -target arm64-apple-macos999 -c %s -### 2>&1 | \
! RUN:   FileCheck --check-prefix=CHECK-MACOS999 %s

! CHECK-MACOS999: "arm64-apple-macosx999.0.0"
