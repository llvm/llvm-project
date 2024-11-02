// Test handling of -f(no-)function-sections and -f(no-)data-sections
//
// CHECK-FS: -ffunction-sections
// CHECK-NOFS-NOT: -ffunction-sections
// CHECK-DS: -fdata-sections
// CHECK-NODS-NOT: -fdata-sections
// CHECK-US-NOT: -fno-unique-section-names
// CHECK-NOUS: -fno-unique-section-names
// CHECK-PLUGIN-DEFAULT-NOT: "-plugin-opt=-function-sections
// CHECK-PLUGIN-DEFAULT-NOT: "-plugin-opt=-data-sections
// CHECK-PLUGIN-SECTIONS: "-plugin-opt=-function-sections=1"
// CHECK-PLUGIN-SECTIONS: "-plugin-opt=-data-sections=1"
// CHECK-PLUGIN-NO-SECTIONS: "-plugin-opt=-function-sections=0"
// CHECK-PLUGIN-NO-SECTIONS: "-plugin-opt=-data-sections=0"

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-NOFS --check-prefix=CHECK-NODS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -ffunction-sections \
// RUN:   | FileCheck --check-prefix=CHECK-FS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fno-function-sections \
// RUN:   | FileCheck --check-prefix=CHECK-NOFS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -ffunction-sections -fno-function-sections \
// RUN:   | FileCheck --check-prefix=CHECK-NOFS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fno-function-sections -ffunction-sections \
// RUN:   | FileCheck --check-prefix=CHECK-FS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -ffunction-sections -fno-function-sections -ffunction-sections \
// RUN:   | FileCheck --check-prefix=CHECK-FS %s


// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fdata-sections \
// RUN:   | FileCheck --check-prefix=CHECK-DS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fno-data-sections \
// RUN:   | FileCheck --check-prefix=CHECK-NODS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fdata-sections -fno-data-sections \
// RUN:   | FileCheck --check-prefix=CHECK-NODS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fno-data-sections -fdata-sections \
// RUN:   | FileCheck --check-prefix=CHECK-DS %s

// RUN: %clang -### %s -fsyntax-only 2>&1       \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fdata-sections -fno-data-sections -fdata-sections \
// RUN:   | FileCheck --check-prefix=CHECK-DS %s


// RUN: %clang -### %s -fsyntax-only 2>&1        \
// RUN:     --target=i386-unknown-linux \
// RUN:     -funique-section-names \
// RUN:   | FileCheck --check-prefix=CHECK-US %s

// RUN: %clang -### %s -fsyntax-only 2>&1        \
// RUN:     --target=i386-unknown-linux \
// RUN:     -fno-unique-section-names \
// RUN:   | FileCheck --check-prefix=CHECK-NOUS %s


// RUN: %clang -### %s -flto 2>&1                \
// RUN:     --target=x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-PLUGIN-DEFAULT %s

// RUN: %clang -### %s -flto 2>&1                \
// RUN:     --target=x86_64-unknown-linux \
// RUN:     -ffunction-sections -fdata-sections \
// RUN:   | FileCheck --check-prefix=CHECK-PLUGIN-SECTIONS %s

// RUN: %clang -### %s -flto 2>&1                \
// RUN:     --target=x86_64-unknown-linux \
// RUN:     -fno-function-sections -fno-data-sections \
// RUN:   | FileCheck --check-prefix=CHECK-PLUGIN-NO-SECTIONS %s
