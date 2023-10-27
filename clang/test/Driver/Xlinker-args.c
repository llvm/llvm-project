// Check that we extract --no-demangle from '-Xlinker' and '-Wl,', since that
// was a collect2 argument.

// RUN: %clang -target i386-apple-darwin9 -### \
// RUN:   -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four -z five -r %s 2> %t
// RUN: FileCheck -check-prefix=DARWIN < %t %s

/// -T is reordered to the last to make sure -L takes precedence.
// RUN: %clang -target x86_64-pc-linux-gnu -### \
// RUN:   -e _start -T a.lds -Xlinker one -Xlinker --no-demangle \
// RUN:   -Wl,two,--no-demangle,three -Xlinker four -z five -r %s 2> %t
// RUN: FileCheck -check-prefix=LINUX < %t %s

/// Check that --no-demangle gets forwarded to the mingw linker
// RUN: %clang -target x86_64-w64-mingw32 -### \
// RUN:   -Wl,--no-demangle %s 2> %t
// RUN: FileCheck -check-prefix=MINGW < %t %s

// RUN: %clang -target powerpc-unknown-aix -### \
// RUN:   -b one -b two %s 2> %t
// RUN: FileCheck -check-prefix=AIX < %t %s

// DARWIN-NOT: --no-demangle
// DARWIN: "one" "two" "three" "four" "-z" "five" "-r"
// LINUX: "--no-demangle" "-e" "_start" "one" "two" "three" "four" "-z" "five" "-r" {{.*}} "-T" "a.lds"
// MINGW: "--no-demangle"
// AIX: "-b" "one" "-b" "two"

// Check that we forward '-Xlinker' and '-Wl,' on Windows.
// RUN: %clang -target i686-pc-win32 -fuse-ld=link -### \
// RUN:   -Xlinker one -Wl,two %s 2>&1 | \
// RUN:   FileCheck -check-prefix=WIN %s
// WIN: link.exe
// WIN: "one"
// WIN: "two"
