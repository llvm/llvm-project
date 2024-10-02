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

// RUN: %clang_cl /c -### /kernel -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-NO-RTTI,KERNEL-NO-EXCEPTIONS %s
// KERNEL-NO-RTTI: "-fno-rtti"
// KERNEL-NO-EXCEPTIONS-NOT: "-fcxx-exceptions" "-fexceptions"

// RUN: not %clang_cl /c -### --target=i686-pc-windows-msvc /kernel /arch:SSE -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-SSE %s
// RUN: not %clang_cl /c -### --target=i686-pc-windows-msvc /kernel /arch:SSE2 -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-SSE2 %s
// RUN: not %clang_cl /c -### --target=i686-pc-windows-msvc /kernel /arch:AVX -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-AVX %s
// RUN: not %clang_cl /c -### --target=i686-pc-windows-msvc /kernel /arch:AVX2 -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-AVX2 %s
// RUN: not %clang_cl /c -### --target=i686-pc-windows-msvc /kernel /arch:AVX512 -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-AVX512 %s
// KERNEL-SSE: error: invalid argument '/arch:SSE' not allowed with '/kernel'
// KERNEL-SSE2: error: invalid argument '/arch:SSE2' not allowed with '/kernel'
// KERNEL-AVX: error: invalid argument '/arch:AVX' not allowed with '/kernel'
// KERNEL-AVX2: error: invalid argument '/arch:AVX2' not allowed with '/kernel'
// KERNEL-AVX512: error: invalid argument '/arch:AVX512' not allowed with '/kernel'

// RUN: %clang_cl /c -### /kernel /EHsc -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-EHSC %s
// RUN: not %clang_cl /c -### /kernel /GR -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-GR %s
// KERNEL-EHSC-NOT: "-fcxx-exceptions" "-fexceptions"
// KERNEL-GR: error: invalid argument '/GR' not allowed with '/kernel'

// RUN: not %clang_cl /c -### --target=x86_64-pc-windows-msvc /kernel /arch:AVX -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-X64-AVX %s
// RUN: not %clang_cl /c -### --target=x86_64-pc-windows-msvc /kernel /arch:AVX2 -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-X64-AVX2 %s
// RUN: not %clang_cl /c -### --target=x86_64-pc-windows-msvc /kernel /arch:AVX512 -- %s 2>&1 | FileCheck -check-prefixes=KERNEL-X64-AVX512 %s
// KERNEL-X64-AVX: error: invalid argument '/arch:AVX' not allowed with '/kernel'
// KERNEL-X64-AVX2: error: invalid argument '/arch:AVX2' not allowed with '/kernel'
// KERNEL-X64-AVX512: error: invalid argument '/arch:AVX512' not allowed with '/kernel'

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
// WCHAR_T-OFF: "-fno-wchar"

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

// thread safe statics are off for versions < 19.
// RUN: %clang_cl /c -### -fms-compatibility-version=18 -- %s 2>&1 | FileCheck -check-prefix=NoThreadSafeStatics %s
// RUN: %clang_cl /Zc:threadSafeInit /Zc:threadSafeInit- /c -### -- %s 2>&1 | FileCheck -check-prefix=NoThreadSafeStatics %s
// NoThreadSafeStatics: "-fno-threadsafe-statics"

// RUN: %clang_cl /Zc:threadSafeInit /c -### -- %s 2>&1 | FileCheck -check-prefix=ThreadSafeStatics %s
// ThreadSafeStatics-NOT: "-fno-threadsafe-statics"

// RUN: %clang_cl /Zc:dllexportInlines- /c -### -- %s 2>&1 | FileCheck -check-prefix=NoDllExportInlines %s
// NoDllExportInlines: "-fno-dllexport-inlines"
// RUN: %clang_cl /Zc:dllexportInlines /c -### -- %s 2>&1 | FileCheck -check-prefix=DllExportInlines %s
// DllExportInlines-NOT: "-fno-dllexport-inlines"

// We recognize -f[no-]delayed-template-parsing.
// /Zc:twoPhase[-] has the opposite meaning.
// RUN: %clang_cl -c -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDDEFAULT %s
// DELAYEDDEFAULT: "-fdelayed-template-parsing"
// RUN: %clang_cl -c -fdelayed-template-parsing -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDON %s
// RUN: %clang_cl -c /Zc:twoPhase- -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDON %s
// DELAYEDON: "-fdelayed-template-parsing"
// RUN: %clang_cl -c -fno-delayed-template-parsing -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDOFF %s
// RUN: %clang_cl -c /Zc:twoPhase -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDOFF %s
// DELAYEDOFF-NOT: "-fdelayed-template-parsing"

// RUN: %clang_cl -c -### /std:c++latest -- %s 2>&1 | FileCheck -check-prefix CHECK-LATEST-CHAR8_T %s
// CHECK-LATEST-CHAR8_T-NOT: "-fchar8_t"
// RUN: %clang_cl -c -### /Zc:char8_t -- %s 2>&1 | FileCheck -check-prefix CHECK-CHAR8_T %s
// CHECK-CHAR8_T: "-fchar8_t"
// RUN: %clang_cl -c -### /Zc:char8_t- -- %s 2>&1 | FileCheck -check-prefix CHECK-CHAR8_T_ %s
// CHECK-CHAR8_T_: "-fno-char8_t"



// These never warn, but don't have an effect yet.

// RUN: %clang_cl /c \
// RUN:   /Zc:__cplusplus \
// RUN:   /Zc:auto \
// RUN:   /Zc:forScope \
// RUN:   /Zc:inline \
// RUN:   /Zc:rvalueCast \
// RUN:   /Zc:ternary \
// RUN:   -### -- %s 2>&1 | FileCheck -DMSG=%errc_ENOENT -check-prefix=IGNORED %s
// IGNORED-NOT: argument unused during compilation
// IGNORED-NOT: [[MSG]]

// Negated form warns:
// RUN: %clang_cl /c \
// RUN:   /Zc:rvalueCast- \
// RUN:   -### -- %s 2>&1 | FileCheck -check-prefix=NOTIGNORED %s
// NOTIGNORED: argument unused during compilation
