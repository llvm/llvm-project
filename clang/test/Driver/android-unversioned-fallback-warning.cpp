// Check that we emit warnings for using unversioned Android target directories
// as appropriate.

// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/include/aarch64-none-linux-android/c++/v1
// RUN: mkdir -p %t/include/aarch64-none-linux-android23/c++/v1
// RUN: mkdir -p %t/include/c++/v1
// RUN: mkdir -p %t/lib/aarch64-none-linux-android
// RUN: mkdir -p %t/lib/aarch64-none-linux-android23
// RUN: mkdir -p %t/resource/lib/aarch64-none-linux-android
// RUN: mkdir -p %t/resource/lib/aarch64-none-linux-android23

// Using an unversioned directory for an unversioned triple isn't a warning.
// RUN: %clang -target aarch64-none-linux-android -ccc-install-dir %t/bin \
// RUN:     -resource-dir %t/resource -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-WARNING %s
// NO-WARNING-NOT: Using unversioned Android target directory

// RUN: %clang -target aarch64-none-linux-android21 -ccc-install-dir %t/bin \
// RUN:     -resource-dir %t/resource -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ANDROID21 -DDIR=%t -DSEP=%{fs-sep} %s
// ANDROID21-DAG: Using unversioned Android target directory [[DIR]]/bin[[SEP]]..[[SEP]]include[[SEP]]aarch64-none-linux-android
// ANDROID21-DAG: Using unversioned Android target directory [[DIR]]/bin[[SEP]]..[[SEP]]lib[[SEP]]aarch64-none-linux-android
// ANDROID21-DAG: Using unversioned Android target directory [[DIR]]/resource[[SEP]]lib[[SEP]]aarch64-none-linux-android

// 23 or newer should use the versioned directory
// RUN: %clang -target aarch64-none-linux-android23 -ccc-install-dir %t/bin \
// RUN:     -resource-dir %t/resource -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-WARNING %s

// RUN: %clang -target aarch64-none-linux-android28 -ccc-install-dir %t/bin \
// RUN:     -resource-dir %t/resource -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-WARNING %s
