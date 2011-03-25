// Check for proper handling of --sysroot and -isysroot flags.

// RUN: %clang -### -fsyntax-only -isysroot /foo/bar %s 2>&1 | \
// RUN:   FileCheck %s -check-prefix=ISYSROOT
// ISYSROOT: "-isysroot" "{{[^"]*}}/foo/bar"

// Check that we get both isysroot for headers, and pass --sysroot on to GCC to
// produce the final binary.
// RUN: %clang -### -ccc-host-triple x86_64-unknown-linux-gnu \
// RUN:   --sysroot=/foo/bar -o /dev/null %s 2>&1 | \
// RUN:   FileCheck %s -check-prefix=SYSROOT_EQ
// SYSROOT_EQ: "-isysroot" "{{[^"]*}}/foo/bar"
// SYSROOT_EQ: "--sysroot{{" "|=}}{{[^"]*}}/foo/bar"

// Check for overriding the header sysroot by providing both --sysroot and
// -isysroot.
// RUN: %clang -### -ccc-host-triple x86_64-unknown-linux-gnu -isysroot /baz \
// RUN:   --sysroot=/foo/bar -o /dev/null %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=ISYSROOT_AND_SYSROOT
// ISYSROOT_AND_SYSROOT: "-isysroot" "{{[^"]*}}/baz"
// ISYSROOT_AND_SYSROOT: "--sysroot{{" "|=}}{{[^"]*}}/foo/bar"

// Check that omitting the equals works as well.
// RUN: %clang -### -ccc-host-triple x86_64-unknown-linux-gnu \
// RUN:   --sysroot /foo/bar -o /dev/null %s 2>&1 | \
// RUN:   FileCheck %s -check-prefix=SYSROOT_SEPARATE
// SYSROOT_SEPARATE: "-isysroot" "{{[^"]*}}/foo/bar"
// SYSROOT_SEPARATE: "--sysroot{{" "|=}}{{[^"]*}}/foo/bar"
