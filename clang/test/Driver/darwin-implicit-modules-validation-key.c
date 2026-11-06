// Check that the Darwin driver adds a validation key for the SDK's
// SystemVersion.plist after the user's keys, and only for implicit modules.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -### --target=arm64-apple-macos14 -isysroot %t/SDK -fmodules \
// RUN:   -fimplicit-modules-validation-key=user -c %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IMPLICIT
// RUN: env -u SDKROOT %clang -### --target=arm64-apple-macos14 --sysroot=%t/SDK -fmodules \
// RUN:   -fimplicit-modules-validation-key=user -c %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IMPLICIT
// IMPLICIT: "-fimplicit-modules-validation-key=user" "-fimplicit-modules-validation-key=darwin-sdk-system-version:{{[0-9a-f]+}}"

// The key changes with the contents of SystemVersion.plist.
// RUN: %clang -### --target=arm64-apple-macos14 -isysroot %t/SDK -fmodules \
// RUN:   -c %t/tu.c 2>&1 | grep -o 'darwin-sdk-system-version:[0-9a-f]*' > %t/key1
// RUN: %clang -### --target=arm64-apple-macos14 -isysroot %t/SDK2 -fmodules \
// RUN:   -c %t/tu.c 2>&1 | grep -o 'darwin-sdk-system-version:[0-9a-f]*' > %t/key2
// RUN: not diff %t/key1 %t/key2

// RUN: %clang -### --target=arm64-apple-macos14 -isysroot %t/SDK -fmodules \
// RUN:   -fno-implicit-modules -fimplicit-modules-validation-key=user \
// RUN:   -c %t/tu.c 2>&1 | FileCheck %s --check-prefix=NONE
// NONE-NOT: warning: argument unused
// NONE: "-cc1"
// NONE-NOT: -fimplicit-modules-validation-key

// No SystemVersion.plist, so only the user's key.
// RUN: %clang -### --target=arm64-apple-macos14 -isysroot %t/NoPlist.sdk \
// RUN:   -fmodules -fimplicit-modules-validation-key=user -c %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=USER-ONLY
// RUN: %clang -### --target=x86_64-linux-gnu -fmodules \
// RUN:   -fimplicit-modules-validation-key=user -c %t/tu.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=USER-ONLY
// USER-ONLY: "-cc1"
// USER-ONLY-SAME: "-fimplicit-modules-validation-key=user"
// USER-ONLY-NOT: darwin-sdk-system-version

//--- SDK/System/Library/CoreServices/SystemVersion.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>ProductBuildVersion</key>
	<string>23A344</string>
	<key>ProductVersion</key>
	<string>14.0</string>
</dict>
</plist>

//--- SDK2/System/Library/CoreServices/SystemVersion.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>ProductBuildVersion</key>
	<string>23B74</string>
	<key>ProductVersion</key>
	<string>14.1</string>
</dict>
</plist>

//--- NoPlist.sdk/usr/include/stdio.h

//--- tu.c
