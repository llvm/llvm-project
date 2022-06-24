// RUN: %clang -### %s -target arm-none-none-eabi -march=armv8a -mfix-cortex-a57-aes-1742098 2>&1 | FileCheck %s --check-prefix=FIX
// RUN: %clang -### %s -target arm-none-none-eabi -march=armv8a -mno-fix-cortex-a57-aes-1742098 2>&1 | FileCheck %s --check-prefix=NO-FIX

// RUN: %clang -### %s -target arm-none-none-eabi -march=armv8a -mfix-cortex-a72-aes-1655431 2>&1 | FileCheck %s --check-prefix=FIX
// RUN: %clang -### %s -target arm-none-none-eabi -march=armv8a -mno-fix-cortex-a72-aes-1655431 2>&1 | FileCheck %s --check-prefix=NO-FIX

// RUN: %clang -### %s -target arm-none-none-eabi -march=armv8a 2>&1 | FileCheck %s --check-prefix=UNSPEC
// RUN: %clang -### %s -target arm-none-none-eabi -march=armv8a 2>&1 | FileCheck %s --check-prefix=UNSPEC

// This test checks that "-m(no-)fix-cortex-a57-aes-1742098" and
// "-m(no-)fix-cortex-a72-aes-1655431" cause the "fix-cortex-a57-aes-1742098"
// target feature to be passed to `clang -cc1`.

// This feature is also enabled in the backend for the two affected CPUs and the
// "generic" cpu (used when only specifying -march), but that won't show up on
// the `clang -cc1` command line.

// We do not check whether this option is correctly specified for the CPU: users
// can specify the "-mfix-cortex-a57-aes-1742098" option with "-mcpu=cortex-a72"
// and vice-versa, and will still get the fix, as the target feature and the fix
// is the same in both cases.

// FIX: "-target-feature" "+fix-cortex-a57-aes-1742098"
// NO-FIX: "-target-feature" "-fix-cortex-a57-aes-1742098"
// UNSPEC-NOT: "-target-feature" "{[+-]}fix-cortex-a57-aes-1742098"
