// Check that we extract --no-demangle from '-Xlinker' and '-Wl,', since that
// was a collect2 argument.

// RUN: %clang -target i386-apple-darwin9 -### \
// RUN:   -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four %s 2> %t
// RUN: FileCheck -check-prefix=DARWIN < %t %s
//
// RUN: %clang -target x86_64-pc-linux-gnu -### \
// RUN:   -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four %s 2> %t
// RUN: FileCheck -check-prefix=LINUX < %t %s
//
// DARWIN-NOT: --no-demangle
// DARWIN: "one" "two" "three" "four"
// LINUX: "--no-demangle" "one" "two" "three" "four"
