// REQUIRES: system-darwin

// Check that ld gets "-lto_library" and warnings about libLTO.dylib path.

// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=133 -flto 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH %s < %t.log
//
// LINK_LTOLIB_PATH: {{ld(.exe)?"}}
// LINK_LTOLIB_PATH: "-lto_library"

// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -ccc-install-dir %S/dummytestdir -mlinker-version=133 -flto 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH_WRN %s < %t.log
//
// LINK_LTOLIB_PATH_WRN: warning: libLTO.dylib relative to clang installed dir not found; using 'ld' default search path instead

// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -ccc-install-dir %S/dummytestdir -mlinker-version=133 -Wno-liblto -flto 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH_NOWRN %s < %t.log
//
// LINK_LTOLIB_PATH_NOWRN-NOT: warning: libLTO.dylib relative to clang installed dir not found; using 'ld' default search path instead
