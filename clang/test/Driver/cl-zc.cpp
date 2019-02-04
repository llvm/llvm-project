// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=TRIGRAPHS-DEFAULT %s
// cc1 will disable trigraphs for -fms-compatibility as long as -ftrigraphs
// isn't explicitly passed.
// TRIGRAPHS-DEFAULT-NOT: "-ftrigraphs"

// RUN: %clang_cl /c -### /Zc:trigraphs -- %s 2>&1 | FileCheck -check-prefix=TRIGRAPHS-ON %s
// TRIGRAPHS-ON: "-ftrigraphs"

// RUN: %clang_cl /c -### /Zc:trigraphs- -- %s 2>&1 | FileCheck -check-prefix=TRIGRAPHS-OFF %s
// TRIGRAPHS-OFF: "-fno-trigraphs"

// RUN: %clang_cl /c -### /Zc:sizedDealloc -- %s 2>&1 | FileCheck -check-prefix=SIZED-DEALLOC-ON %s
// SIZED-DEALLOC-ON: "-fsized-deallocation"

// RUN: %clang_cl /c -### /Zc:sizedDealloc- -- %s 2>&1 | FileCheck -check-prefix=SIZED-DEALLOC-OFF %s
// SIZED-DEALLOC-OFF-NOT: "-fsized-deallocation"

// RUN: %clang_cl /c /std:c++17 -### /Zc:alignedNew -- %s 2>&1 | FileCheck -check-prefix=ALIGNED-NEW-ON %s
// ALIGNED-NEW-ON: "-faligned-allocation"

// RUN: %clang_cl /c /std:c++17 -### /Zc:alignedNew- -- %s 2>&1 | FileCheck -check-prefix=ALIGNED-NEW-OFF %s
// ALIGNED-NEW-OFF-NOT: "-faligned-allocation"

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=STRICTSTRINGS-DEFAULT %s
// STRICTSTRINGS-DEFAULT-NOT: -Werror=c++11-compat-deprecated-writable-strings
// RUN: %clang_cl /c -### /Zc:strictStrings -- %s 2>&1 | FileCheck -check-prefix=STRICTSTRINGS-ON %s
// STRICTSTRINGS-ON: -Werror=c++11-compat-deprecated-writable-strings
// RUN: %clang_cl /c -### /Zc:strictStrings- -- %s 2>&1 | FileCheck -check-prefix=STRICTSTRINGS-OFF %s
// STRICTSTRINGS-OFF: argument unused during compilation


// RUN: %clang_cl /c -### /Zc:foobar -- %s 2>&1 | FileCheck -check-prefix=FOOBAR-ON %s
// FOOBAR-ON: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:foobar- -- %s 2>&1 | FileCheck -check-prefix=FOOBAR-ON %s
// FOOBAR-OFF: argument unused during compilation

// These are ignored if enabled, and warn if disabled.

// RUN: %clang_cl /c -### /Zc:forScope -- %s 2>&1 | FileCheck -check-prefix=FORSCOPE-ON %s
// FORSCOPE-ON-NOT: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:forScope- -- %s 2>&1 | FileCheck -check-prefix=FORSCOPE-OFF %s
// FORSCOPE-OFF: argument unused during compilation

// RUN: %clang_cl /c -### /Zc:wchar_t -- %s 2>&1 | FileCheck -check-prefix=WCHAR_T-ON %s
// WCHAR_T-ON-NOT: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:wchar_t- -- %s 2>&1 | FileCheck -check-prefix=WCHAR_T-OFF %s
// WCHAR_T-OFF: argument unused during compilation

// RUN: %clang_cl /c -### /Zc:auto -- %s 2>&1 | FileCheck -check-prefix=AUTO-ON %s
// AUTO-ON-NOT: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:auto- -- %s 2>&1 | FileCheck -check-prefix=AUTO-OFF %s
// AUTO-OFF: argument unused during compilation

// RUN: %clang_cl /c -### /Zc:inline -- %s 2>&1 | FileCheck -check-prefix=INLINE-ON %s
// INLINE-ON-NOT: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:inline- -- %s 2>&1 | FileCheck -check-prefix=INLINE-OFF %s
// INLINE-OFF: argument unused during compilation

// RUN: %clang_cl /c -### /Zc:ternary -- %s 2>&1 | FileCheck -check-prefix=TERNARY-ON %s
// TERNARY-ON-NOT: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:ternary- -- %s 2>&1 | FileCheck -check-prefix=TERNARY-OFF %s
// TERNARY-OFF: argument unused during compilation


// These never warn, but don't have an effect yet.

// RUN: %clang_cl /c -### /Zc:rvalueCast -- %s 2>&1 | FileCheck -check-prefix=RVALUECAST-ON %s
// RVALUECAST-ON-NOT: argument unused during compilation
// RUN: %clang_cl /c -### /Zc:rvalueCast- -- %s 2>&1 | FileCheck -check-prefix=RVALUECAST-OFF %s
// RVALUECAST-OFF: argument unused during compilation
