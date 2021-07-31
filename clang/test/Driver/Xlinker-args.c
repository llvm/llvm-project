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

// RUN: %clang -target powerpc-unknown-aix -### \
// RUN:   -b one %s 2> %t
// RUN: FileCheck -check-prefix=AIX < %t %s

// RUN: %clang -target powerpc-unknown-linux -### \
// RUN:   -b one %s 2> %t
// RUN: FileCheck -check-prefix=NOT-AIX < %t %s

// DARWIN-NOT: --no-demangle
// DARWIN: "one" "two" "three" "four" "-z" "five" "-r"
// LINUX: "--no-demangle" "-e" "_start" "one" "two" "three" "four" "-z" "five" "-r" {{.*}} "-T" "a.lds"
// AIX: "-b" "one" 
// NOT-AIX: error: unsupported option '-b' for target 'powerpc-unknown-linux'

// Check that we forward '-Xlinker' and '-Wl,' on Windows.
// RUN: %clang -target i686-pc-win32 -fuse-ld=link -### \
// RUN:   -Xlinker one -Wl,two %s 2>&1 | \
// RUN:   FileCheck -check-prefix=WIN %s
// WIN: link.exe
// WIN: "one"
// WIN: "two"
