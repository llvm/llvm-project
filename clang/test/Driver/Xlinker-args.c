// Check that we extract --no-demangle from '-Xlinker' and '-Wl,', since that
// was a collect2 argument.

// RUN: %clang -target i386-apple-darwin9 -### \
// RUN:   -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four -z five -r %s 2> %t
// RUN: FileCheck -check-prefix=DARWIN < %t %s
//
// RUN: %clang -target x86_64-pc-linux-gnu -### \
// RUN:   -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four -z five -r %s 2> %t
// RUN: FileCheck -check-prefix=LINUX < %t %s
//
// DARWIN-NOT: --no-demangle
// DARWIN: "one" "two" "three" "four" "-z" "five" "-r"
// LINUX: "--no-demangle" "one" "two" "three" "four" "-z" "five" "-r"

// Check that we forward '-Xlinker' and '-Wl,' on Windows.
// RUN: %clang -target i686-pc-win32 -### \
// RUN:   -Xlinker one -Wl,two %s 2>&1 | \
// RUN:   FileCheck -check-prefix=WIN %s
// WIN: link.exe
// WIN: "one"
// WIN: "two"
