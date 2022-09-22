// Check that files that should not impact the module build are not included in
// the cas-fs, which can cause spurious rebuilds.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_cache2.json.template > %t/cdb_cache2.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_timestamp.json.template > %t/cdb_timestamp.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Changing module cache path should not affect results.
// RUN: clang-scan-deps -compilation-database %t/cdb_cache2.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_cache2.json
// RUN: diff -u %t/deps_cache2.json %t/deps.json

// .pcm.timestamp files created by -fmodules-validate-once-per-build-session should not affect results
// RUN: touch %t/session
// RUN: clang-scan-deps -compilation-database %t/cdb_timestamp.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pre_timestamp.json
// RUN: touch %t/Top.h
// RUN: clang-scan-deps -compilation-database %t/cdb_timestamp.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_post_timestamp.json
// RUN: diff -u %t/deps_pre_timestamp.json %t/deps_post_timestamp.json

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang_tool -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache",
  "file" : "DIR/tu.c"
}]

//--- cdb_cache2.json.template
[{
  "directory" : "DIR",
  "command" : "clang_tool -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache2 -Rcompile-job-cache",
  "file" : "DIR/tu.c"
}]

//--- cdb_timestamp.json.template
[{
  "directory" : "DIR",
  "command" : "clang_tool -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache2 -fmodules-validate-once-per-build-session -fbuild-session-file=DIR/session -Rcompile-job-cache",
  "file" : "DIR/tu.c"
}]

//--- module.modulemap
module Top { header "Top.h" export * }
module Left { header "Left.h" export * }
module Right { header "Right.h" export * }

//--- Top.h
#pragma once
void Top(void);

//--- Left.h
#pragma once
#include "Top.h"
void Left(void);

//--- Right.h
#pragma once
#include "Top.h"
void Right(void);

//--- tu.c
#include "Left.h"
#include "Right.h"

void tu(void) {
  Top();
  Left();
  Right();
}
