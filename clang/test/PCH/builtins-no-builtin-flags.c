// Whether an identifier names a builtin follows the -fno-builtin-* options of
// the compilation that imports a PCH, as for modules, even if the PCH records
// the identifier. A function declared by the PCH keeps whether it was declared
// as a builtin, though.

// RUN: rm -rf %t
// RUN: split-file %s %t

// The PCH only mentions strlen in a macro.
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-pch -o %t/mentions.pch \
// RUN:   %t/mentions.h
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-pch \
// RUN:   -o %t/mentions-no-strlen.pch -fno-builtin-strlen %t/mentions.h
// RUN: %clang_cc1 -triple arm64-apple-macosx -include-pch %t/mentions.pch \
// RUN:   -fsyntax-only -verify=builtin %t/tu.c
// RUN: %clang_cc1 -triple arm64-apple-macosx -include-pch %t/mentions.pch \
// RUN:   -fno-builtin-strlen -fsyntax-only -verify=no-builtin %t/tu.c
// RUN: %clang_cc1 -triple arm64-apple-macosx \
// RUN:   -include-pch %t/mentions-no-strlen.pch -fsyntax-only -verify=builtin \
// RUN:   %t/tu.c

// The PCH declares strlen.
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-pch -o %t/declares.pch \
// RUN:   %t/declares.h
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-pch \
// RUN:   -o %t/declares-no-strlen.pch -fno-builtin-strlen %t/declares.h
// RUN: %clang_cc1 -triple arm64-apple-macosx -include-pch %t/declares.pch \
// RUN:   -fsyntax-only -verify=builtin -DDECLARED %t/tu.c
// RUN: %clang_cc1 -triple arm64-apple-macosx \
// RUN:   -include-pch %t/declares-no-strlen.pch -fsyntax-only \
// RUN:   -verify=no-builtin -DDECLARED %t/tu.c

//--- mentions.h
#define LEN(S) strlen(S)

//--- declares.h
#define LEN(S) strlen(S)
unsigned long strlen(const char *);

//--- tu.c
// builtin-no-diagnostics
#ifndef DECLARED
unsigned long strlen(const char *);
#endif
// no-builtin-error@+1 {{static assertion expression is not an integral constant expression}}
_Static_assert(LEN("abc") == 3, "");
