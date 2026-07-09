// RUN: %clang -### -target arm64-apple-macosx11 -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -target arm64-apple-macosx11 -fobjc-constant-literals -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -target arm64-apple-macosx11 -fno-objc-constant-literals -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLED

// DEFAULT: -fobjc-constant-literals
// DEFAULT: -fconstant-nsnumber-literals
// DEFAULT: -fconstant-nsarray-literals
// DEFAULT: -fconstant-nsdictionary-literals

// ENABLED: -fobjc-constant-literals
// ENABLED: -fconstant-nsnumber-literals
// ENABLED: -fconstant-nsarray-literals
// ENABLED: -fconstant-nsdictionary-literals

// DISABLED-NOT: -fobjc-constant-literals
// DISABLED-NOT: -fconstant-nsnumber-literals
// DISABLED-NOT: -fconstant-nsarray-literals
// DISABLED-NOT: -fconstant-nsdictionary-literals

// The constant literal flags are Objective-C only. A build system may still
// pass them uniformly to non-ObjC inputs (e.g. assembly files via a shared
// response file); they carry NoArgumentUnused so they don't trigger
// -Wunused-command-line-argument under -Werror.
// RUN: %clang -### -target arm64-apple-macosx11 -Werror \
// RUN:   -fobjc-constant-literals -fconstant-nsnumber-literals \
// RUN:   -fconstant-nsarray-literals -fconstant-nsdictionary-literals \
// RUN:   -x assembler -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IGNORED --allow-empty
// RUN: %clang -### -target arm64-apple-macosx11 -Werror \
// RUN:   -fno-objc-constant-literals -fno-constant-nsnumber-literals \
// RUN:   -fno-constant-nsarray-literals -fno-constant-nsdictionary-literals \
// RUN:   -x assembler -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IGNORED --allow-empty

// IGNORED-NOT: argument unused during compilation
// IGNORED-NOT: -fobjc-constant-literals
// IGNORED-NOT: -fconstant-nsnumber-literals
// IGNORED-NOT: -fconstant-nsarray-literals
// IGNORED-NOT: -fconstant-nsdictionary-literals

// The same flags must also be ignored (and not forwarded to cc1) when compiling
// non-ObjC C/C++ inputs.
// RUN: %clang -### -target arm64-apple-macosx11 -Werror \
// RUN:   -fobjc-constant-literals -fconstant-nsnumber-literals \
// RUN:   -fconstant-nsarray-literals -fconstant-nsdictionary-literals \
// RUN:   -x c++ -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IGNORED --allow-empty
