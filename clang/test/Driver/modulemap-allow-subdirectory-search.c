// RUN: rm -rf %t
// RUN: split-file %s %t

// Check that with a sufficiently new SDK not searching for module maps in subdirectories.

// New SDK.
// RUN: %clang -target x86_64-apple-macos10.13 -isysroot %t/MacOSX15.0.sdk -fmodules %t/test.c -### 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-SUBDIRECTORIES %t/test.c
// Old SDK.
// RUN: %clang -target x86_64-apple-macos10.13 -isysroot %t/MacOSX14.0.sdk -fmodules %t/test.c -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SEARCH-SUBDIRECTORIES %t/test.c
// Non-Darwin platform.
// RUN: %clang -target i386-unknown-linux -isysroot %t/MacOSX15.0.sdk -fmodules %t/test.c -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SEARCH-SUBDIRECTORIES %t/test.c
// New SDK overriding the default.
// RUN: %clang -target x86_64-apple-macos10.13 -isysroot %t/MacOSX15.0.sdk -fmodules %t/test.c -fmodulemap-allow-subdirectory-search -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SEARCH-SUBDIRECTORIES %t/test.c

//--- test.c
// NO-SUBDIRECTORIES: "-fno-modulemap-allow-subdirectory-search"
// SEARCH-SUBDIRECTORIES-NOT: "-fno-modulemap-allow-subdirectory-search"

//--- MacOSX15.0.sdk/SDKSettings.json
{"Version":"15.0", "MaximumDeploymentTarget": "15.0.99"}

//--- MacOSX14.0.sdk/SDKSettings.json
{"Version":"14.0", "MaximumDeploymentTarget": "14.0.99"}
