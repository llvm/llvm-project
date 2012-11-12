// Check that ld gets arch_multiple.

// RUN: %clang -target i386-apple-darwin9 -arch i386 -arch x86_64 %s -### -o foo 2> %t.log
// RUN: grep '".*ld.*" .*"-arch_multiple" "-final_output" "foo"' %t.log

// Make sure we run dsymutil on source input files.
// RUN: %clang -target i386-apple-darwin9 -### -g %s -o BAR 2> %t.log
// RUN: grep '".*dsymutil" "-o" "BAR.dSYM" "BAR"' %t.log
// RUN: %clang -target i386-apple-darwin9 -### -g -filelist FOO %s -o BAR 2> %t.log
// RUN: grep '".*dsymutil" "-o" "BAR.dSYM" "BAR"' %t.log

// Check linker changes that came with new linkedit format.
// RUN: touch %t.o
// RUN: %clang -target i386-apple-darwin9 -### -arch armv6 -miphoneos-version-min=3.0 %t.o 2> %t.log
// RUN: %clang -target i386-apple-darwin9 -### -arch armv6 -miphoneos-version-min=3.0 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -target i386-apple-darwin9 -### -arch armv6 -miphoneos-version-min=3.0 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONE_3_0 %s < %t.log

// LINK_IPHONE_3_0: {{ld(.exe)?"}}
// LINK_IPHONE_3_0-NOT: -lcrt1.3.1.o
// LINK_IPHONE_3_0: -lcrt1.o
// LINK_IPHONE_3_0: -lSystem
// LINK_IPHONE_3_0: {{ld(.exe)?"}}
// LINK_IPHONE_3_0: -dylib
// LINK_IPHONE_3_0: -ldylib1.o
// LINK_IPHONE_3_0: -lSystem
// LINK_IPHONE_3_0: {{ld(.exe)?"}}
// LINK_IPHONE_3_0: -lbundle1.o
// LINK_IPHONE_3_0: -lSystem

// RUN: %clang -target i386-apple-darwin9 -### -arch armv7 -miphoneos-version-min=3.1 %t.o 2> %t.log
// RUN: %clang -target i386-apple-darwin9 -### -arch armv7 -miphoneos-version-min=3.1 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -target i386-apple-darwin9 -### -arch armv7 -miphoneos-version-min=3.1 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONE_3_1 %s < %t.log

// LINK_IPHONE_3_1: {{ld(.exe)?"}}
// LINK_IPHONE_3_1-NOT: -lcrt1.o
// LINK_IPHONE_3_1: -lcrt1.3.1.o
// LINK_IPHONE_3_1: -lSystem
// LINK_IPHONE_3_1: {{ld(.exe)?"}}
// LINK_IPHONE_3_1: -dylib
// LINK_IPHONE_3_1-NOT: -ldylib1.o
// LINK_IPHONE_3_1: -lSystem
// LINK_IPHONE_3_1: {{ld(.exe)?"}}
// LINK_IPHONE_3_1-NOT: -lbundle1.o
// LINK_IPHONE_3_1: -lSystem

// RUN: %clang -target i386-apple-darwin9 -### -fpie %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_EXPLICIT_PIE %s < %t.log
//
// LINK_EXPLICIT_PIE: {{ld(.exe)?"}}
// LINK_EXPLICIT_PIE: "-pie"

// RUN: %clang -target i386-apple-darwin9 -### -fno-pie %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_EXPLICIT_NO_PIE %s < %t.log
//
// LINK_EXPLICIT_NO_PIE: {{ld(.exe)?"}}
// LINK_EXPLICIT_NO_PIE: "-no_pie"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -mlinker-version=100 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NEWER_DEMANGLE %s < %t.log
//
// LINK_NEWER_DEMANGLE: {{ld(.exe)?"}}
// LINK_NEWER_DEMANGLE: "-demangle"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -mlinker-version=100 -Wl,--no-demangle 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NEWER_NODEMANGLE %s < %t.log
//
// LINK_NEWER_NODEMANGLE: {{ld(.exe)?"}}
// LINK_NEWER_NODEMANGLE-NOT: "-demangle"
// LINK_NEWER_NODEMANGLE: "-lSystem"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -mlinker-version=95 2> %t.log
// RUN: FileCheck -check-prefix=LINK_OLDER_NODEMANGLE %s < %t.log
//
// LINK_OLDER_NODEMANGLE: {{ld(.exe)?"}}
// LINK_OLDER_NODEMANGLE-NOT: "-demangle"
// LINK_OLDER_NODEMANGLE: "-lSystem"

// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -mlinker-version=117 -flto 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=LINK_OBJECT_LTO_PATH %s < %t.log
//
// LINK_OBJECT_LTO_PATH: {{ld(.exe)?"}}
// LINK_OBJECT_LTO_PATH: "-object_path_lto"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -force_load a -force_load b 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=FORCE_LOAD %s < %t.log
//
// FORCE_LOAD: {{ld(.exe)?"}}
// FORCE_LOAD: "-force_load" "a" "-force_load" "b"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -lazy_framework Framework 2> %t.log
//
// RUN: FileCheck -check-prefix=LINK_LAZY_FRAMEWORK %s < %t.log
// LINK_LAZY_FRAMEWORK: {{ld(.exe)?"}}
// LINK_LAZY_FRAMEWORK: "-lazy_framework" "Framework"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -lazy_library Library 2> %t.log
//
// RUN: FileCheck -check-prefix=LINK_LAZY_LIBRARY %s < %t.log
// LINK_LAZY_LIBRARY: {{ld(.exe)?"}}
// LINK_LAZY_LIBRARY: "-lazy_library" "Library"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o 2> %t.log
// RUN: %clang -target x86_64-apple-macosx10.7 -### %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_VERSION_MIN %s < %t.log
// LINK_VERSION_MIN: {{ld(.exe)?"}}
// LINK_VERSION_MIN: "-macosx_version_min" "10.6.0"
// LINK_VERSION_MIN: {{ld(.exe)?"}}
// LINK_VERSION_MIN: "-macosx_version_min" "10.7.0"

// RUN: %clang -target x86_64-apple-darwin12 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_CRT1 %s < %t.log
// LINK_NO_CRT1-NOT: crt

// RUN: %clang -target armv7-apple-ios6.0 -miphoneos-version-min=6.0 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_IOS_CRT1 %s < %t.log
// LINK_NO_IOS_CRT1-NOT: crt

// RUN: %clang -target i386-apple-darwin12 -pg -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_PG %s < %t.log
// LINK_PG: -lgcrt1.o
// LINK_PG: -no_new_main
