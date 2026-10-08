// Test the negative dependency of adding a header to a directory that is
// searched before the one an include previously resolved in. Nothing the first
// scan recorded as an input changes, only the listing of that directory.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// 0) The search directories are checked when the scanning module is loaded,
//    so the tracking scans below can reuse one built by a scan that doesn't
//    track them.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full > /dev/null

// 1) A clean scan resolves both headers in inc/B. Only the translation unit
//    reports the search directories.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -track-search-directories 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=CLEAN

// CLEAN-NOT:  "directory-deps"
// CLEAN:      "file-deps": [
// CLEAN-NEXT:   "[[PREFIX]]/inc/Mod/module.modulemap",
// CLEAN-NEXT:   "[[PREFIX]]/inc/Mod/Mod.h",
// CLEAN-NEXT:   "[[PREFIX]]/inc/B/Value.h",
// CLEAN-NEXT:   "[[PREFIX]]/inc/B/Other.h"
// CLEAN-NEXT: ]
// CLEAN:      "name": "Mod"
// CLEAN:      "directory-deps": [
// CLEAN-NEXT:   "[[PREFIX]]/inc/A",
// CLEAN-NEXT:   "[[PREFIX]]/inc/B",
// CLEAN-NEXT:   "[[PREFIX]]/inc/Mod"
// CLEAN-NEXT: ]

// 2) Shadow one of the headers from the directory that is searched first.
//    Module files written in the second a scan starts count as up to date with
//    it, so leave a second between builds.
// RUN: sleep 1
// RUN: touch %t/inc/A/Value.h

// 3) Re-scan without -invalidated-path: the cached module is reused, so
//    the stale resolution is reported again. This is the negative dependency
//    the build system has to close.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -track-search-directories 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=STALE

// STALE:     "[[PREFIX]]/inc/B/Value.h"
// STALE-NOT: "[[PREFIX]]/inc/A/Value.h"

// 4) Re-scan with -invalidated-path: the scanning module shares the
//    translation unit's header search, so it is rebuilt and picks up the
//    shadowing header.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -track-search-directories \
// RUN:   -invalidated-path=%/t/inc/A 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=FRESH

// FRESH:      "file-deps": [
// FRESH-NEXT:   "[[PREFIX]]/inc/Mod/module.modulemap",
// FRESH-NEXT:   "[[PREFIX]]/inc/Mod/Mod.h",
// FRESH-NEXT:   "[[PREFIX]]/inc/A/Value.h",
// FRESH-NEXT:   "[[PREFIX]]/inc/B/Other.h"
// FRESH-NEXT: ]
// FRESH:      "name": "Mod"

//--- inc/A/Placeholder.h
//--- inc/B/Value.h
//--- inc/B/Other.h

//--- inc/Mod/module.modulemap
module Mod { header "Mod.h" }

//--- inc/Mod/Mod.h
#include "Value.h"
#include "Other.h"

//--- tu.c
#include "Mod.h"

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR/inc/A -I DIR/inc/B -I DIR/inc/Mod -c DIR/tu.c -o DIR/tu.o"
}]

