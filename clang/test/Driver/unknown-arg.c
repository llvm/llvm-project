// RUN: not %clang %s -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -ifoo -imultilib dir -### 2>&1 | \
// RUN: FileCheck %s
// RUN: %clang %s -imultilib dir -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=MULTILIB
// RUN: not %clang %s -stdlibs=foo -hell -version -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -### -c -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL
// RUN: %clang_cl -Brepo -### -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL-DID-YOU-MEAN
// RUN: not %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -Werror=unknown-argument -### -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL-ERROR
// RUN: not %clang_cl -helo -Werror=unknown-argument -### -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=CL-ERROR-DID-YOU-MEAN
// RUN: %clang_cl -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats -funknown-to-clang-option -c -Wno-unknown-argument -### -- %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=SILENT
// RUN: not %clang -cc1as -hell --version -debug-info-macros 2>&1 | \
// RUN: FileCheck %s --check-prefix=CC1AS-DID-YOU-MEAN
// RUN: not %clang -cc1asphalt -help 2>&1 | \
// RUN: FileCheck %s --check-prefix=UNKNOWN-INTEGRATED

// CHECK: error: unknown argument: '-cake-is-lie'
// CHECK: error: unknown argument: '-%0'
// CHECK: error: unknown argument: '-%d'
// CHECK: error: unknown argument: '-HHHH'
// CHECK: error: unknown argument: '-munknown-to-clang-option'
// CHECK: error: unknown argument: '-print-stats'
// CHECK: error: unknown argument: '-funknown-to-clang-option'
// CHECK: error: unknown argument: '-ifoo'
// MULTILIB: warning: argument unused during compilation: '-imultilib dir'
// DID-YOU-MEAN: error: unknown argument '-stdlibs=foo', did you mean '-stdlib=foo'?
// DID-YOU-MEAN: error: unknown argument '-hell', did you mean '-help'?
// DID-YOU-MEAN: error: unknown argument '-version', did you mean '--version'?
// CL: warning: unknown argument ignored in clang-cl: '-cake-is-lie'
// CL: warning: unknown argument ignored in clang-cl: '-%0'
// CL: warning: unknown argument ignored in clang-cl: '-%d'
// CL: warning: unknown argument ignored in clang-cl: '-HHHH'
// CL: warning: unknown argument ignored in clang-cl: '-munknown-to-clang-option'
// CL: warning: unknown argument ignored in clang-cl: '-print-stats'
// CL: warning: unknown argument ignored in clang-cl: '-funknown-to-clang-option'
// CL-DID-YOU-MEAN: warning: unknown argument ignored in clang-cl '-Brepo' (did you mean '-Brepro'?)
// CL-ERROR: error: unknown argument ignored in clang-cl: '-cake-is-lie'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-%0'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-%d'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-HHHH'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-munknown-to-clang-option'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-print-stats'
// CL-ERROR: error: unknown argument ignored in clang-cl: '-funknown-to-clang-option'
// CL-ERROR-DID-YOU-MEAN: error: unknown argument ignored in clang-cl '-helo' (did you mean '-help'?)
// SILENT-NOT: error:
// SILENT-NOT: warning:
// CC1AS-DID-YOU-MEAN: error: unknown argument '-hell', did you mean '-help'?
// CC1AS-DID-YOU-MEAN: error: unknown argument '--version', did you mean '-version'?
// UNKNOWN-INTEGRATED: error: unknown integrated tool 'asphalt'. Valid tools include '-cc1' and '-cc1as'.

// RUN: %clang -S %s -o %t.s  -Wunknown-to-clang-option 2>&1 | FileCheck --check-prefix=IGNORED %s

// IGNORED: warning: unknown warning option '-Wunknown-to-clang-option'
