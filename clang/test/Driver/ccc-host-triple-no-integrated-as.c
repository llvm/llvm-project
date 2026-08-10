// Check that -no-integrated-as works when -target i386-pc-win32-macho or
// -target x86_64-pc-win32-macho is specified.

// RUN: %clang -### -c -target i386-pc-win32-macho -no-integrated-as %s 2> %t1
// RUN: FileCheck -check-prefix=X86 < %t1 %s
// RUN: %clang -### -c -target x86_64-pc-win32-macho -no-integrated-as %s 2> %t2
// RUN: FileCheck -check-prefix=X86_64 < %t2 %s
//
// X86: "-cc1"
// X86-NOT: "-cc1as"
// X86: "-arch"
// X86: "i386"
//
// X86_64: "-cc1"
// X86_64-NOT: "-cc1as"
// X86_64: "-arch"
// X86_64: "x86_64"
