// FIXME: Disable pending PR4941.
// RUX: clang -target x86_64-apple-darwin9 -arch arm -### -fsyntax-only %s 2> %t &&
// RUX: grep -- "-fno-builtin-strcat" %t &&
// RUX: grep -- "-fno-builtin-strcpy" %t &&

// FIXME: Disable pending PR4941.
// RUX: clang -target x86_64-apple-darwin9 -arch arm -### -fsyntax-only %s -fbuiltin-strcat -fbuiltin-strcpy 2> %t &&
// RUX: not grep -- "-fno-builtin-strcat" %t &&
// RUX: not grep -- "-fno-builtin-strcpy" %t &&

// RUN: %clang -target x86_64-apple-darwin9 -arch arm -### -fsyntax-only %s -fbuiltin-strcat -fbuiltin-strcpy 2> %t
// RUN: not grep -- "-fno-builtin-strcat" %t
// RUN: not grep -- "-fno-builtin-strcpy" %t

