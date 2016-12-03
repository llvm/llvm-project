// REQUIRES: system-darwin

// Check that ld gets "-lto_library".

// RUN: mkdir -p %T/bin
// RUN: mkdir -p %T/lib
// RUN: touch %T/lib/libLTO.dylib
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -ccc-install-dir %T/bin -mlinker-version=133 2> %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH %s -input-file %t.log
//
// LINK_LTOLIB_PATH: {{ld(.exe)?"}}
// LINK_LTOLIB_PATH: "-lto_library"

// Also pass -lto_library even if the file doesn't exist; if it's needed at
// link time, ld will complain instead.
// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -ccc-install-dir %S/dummytestdir -mlinker-version=133 2> %t.log
// RUN: FileCheck -check-prefix=LINK_LTOLIB_PATH %s -input-file %t.log
