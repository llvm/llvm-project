// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 \
// RUN:   subdir/tu.c -I. -fdiagnostics-absolute-paths -E \
// RUN:   -Rcompile-job-cache > %t/stdout-miss 2> %t/stderr-miss
// RUN: cat %t/stdout-miss | %PathSanitizingFileCheck %s --sanitize PREFIX=%/t -check-prefix=PP
// RUN: cat %t/stderr-miss | %PathSanitizingFileCheck %s --sanitize PREFIX=%/t -check-prefix=DIAG
// RUN: cat %t/stderr-miss | %PathSanitizingFileCheck %s --sanitize PREFIX=%/t -check-prefix=MISS

// Again, but with a cache hit.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 \
// RUN:   subdir/tu.c -I. -fdiagnostics-absolute-paths -E \
// RUN:   -Rcompile-job-cache > %t/stdout-hit 2> %t/stderr-hit
// RUN: cat %t/stdout-hit | %PathSanitizingFileCheck %s --sanitize PREFIX=%/t -check-prefix=PP
// RUN: cat %t/stderr-hit | %PathSanitizingFileCheck %s --sanitize PREFIX=%/t -check-prefix=DIAG
// RUN: cat %t/stderr-hit | %PathSanitizingFileCheck %s --sanitize PREFIX=%/t -check-prefix=HIT

// Preprocessor does not force absolute paths.
// PP: # 1 "subdir{{/|\\\\}}tu.c"
// PP: # 1 ".{{/|\\\\}}h1.h"
// PP: const char *filename = ".{{/|\\\\}}h1.h";

// Diagnostics force absolute paths.
// DIAG: PREFIX{{/|\\}}subdir{{/|\\}}tu.c:1:2: warning: "from tu"
// DIAG: In file included from PREFIX{{/|\\}}subdir{{/|\\}}tu.c
// DIAG: PREFIX{{/|\\}}h1.h:1:2: warning: "from h1"

// MISS: remark: compile job cache miss
// HIT: remark: compile job cache hit

//--- subdir/tu.c
#warning "from tu"
#include "h1.h"

//--- h1.h
#warning "from h1"
const char *filename = __FILE__;
