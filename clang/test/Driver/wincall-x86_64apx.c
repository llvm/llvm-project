// x86_64apx-windows targets default to the wincall calling convention and pass
// a 64 KiB section alignment to the linker.

// RUN: %clang --target=x86_64apx-unknown-windows-msvc -### %s 2>&1 | FileCheck --check-prefix=MSVC-APX %s
// RUN: %clang --target=x86_64apx-unknown-windows-gnu -### %s 2>&1 | FileCheck --check-prefix=GNU-APX %s

// Non-APX Windows targets do not get the section alignment flag.
// RUN: %clang --target=x86_64-unknown-windows-msvc -### %s 2>&1 | FileCheck --check-prefix=MSVC-PLAIN %s
// RUN: %clang --target=x86_64-unknown-windows-gnu -### %s 2>&1 | FileCheck --check-prefix=GNU-PLAIN %s

// MSVC-APX: link.exe"
// MSVC-APX-SAME: "-nologo" "/section-alignment:0x10000" "/driver"
// MSVC-PLAIN-NOT: section-alignment
// GNU-APX: "--section-alignment=0x10000"
// GNU-PLAIN-NOT: section-alignment

// User-provided -Wl section-alignment overrides the default.
// RUN: %clang --target=x86_64apx-unknown-windows-msvc -Wl,/section-alignment:0x20000 -### %s 2>&1 | FileCheck --check-prefix=MSVC-OVERRIDE %s
// RUN: %clang --target=x86_64apx-unknown-windows-gnu -Wl,--section-alignment=0x20000 -### %s 2>&1 | FileCheck --check-prefix=GNU-OVERRIDE %s
// MSVC-OVERRIDE-NOT: "/section-alignment:0x10000"
// MSVC-OVERRIDE: "/section-alignment:0x20000"
// GNU-OVERRIDE-NOT: --section-alignment=0x10000
// GNU-OVERRIDE: --section-alignment=0x20000
