// Test directory dependency tracking. This handles the negative dependency case
// of adding a header to an umbrella directory between incremental scans.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

//--- include/Umb/module.modulemap
module UmbDir {
  umbrella "sub"
  module * { export * }
}
module UmbHdr {
  umbrella header "umb.h"
}

//--- include/Umb/sub/a.h
//--- include/Umb/umb.h

//--- tu.c
#include "Umb/sub/a.h"
#include "Umb/umb.h"

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR/include -c DIR/tu.c -o DIR/tu.o"
}]

// 1) A clean scan reports the enumerated umbrella directory for UmbDir, and the
//    umbrella header's directory for UmbHdr.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full 2>&1 | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=CLEAN

// CLEAN:      "directory-deps": [
// CLEAN-NEXT:   "[[PREFIX]]/include/Umb/sub"
// CLEAN-NEXT: ]
// CLEAN:      "file-deps": [
// CLEAN-NEXT:   "[[PREFIX]]/include/Umb/module.modulemap"
// CLEAN-NEXT:   "[[PREFIX]]/include/Umb/sub/a.h"
// CLEAN-NEXT: ]
// CLEAN:      "name": "UmbDir"
// CLEAN:      "directory-deps": [
// CLEAN-NEXT:   "[[PREFIX]]/include/Umb"
// CLEAN-NEXT: ]
// CLEAN:      "name": "UmbHdr"

// 2) The first scan populated the module cache. Add a header to the enumerated
//    directory.
// RUN: touch %t/include/Umb/sub/b.h

// 3) Re-scan without -invalidated-directory: the cached module is reused, so
//    b.h is not picked up. This is the negative dependency the build system has
//    to close, and is why the field is reported at all.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full 2>&1 | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=STALE

// STALE:      "file-deps": [
// STALE-NEXT:   "[[PREFIX]]/include/Umb/module.modulemap"
// STALE-NEXT:   "[[PREFIX]]/include/Umb/sub/a.h"
// STALE-NEXT: ]
// STALE:      "name": "UmbDir"
// STALE-NOT:  "[[PREFIX]]/include/Umb/sub/b.h"

// 4) Re-scan with -invalidated-directory: the module that enumerated the
//    changed directory is out of date and rebuilt, so b.h now appears.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full \
// RUN:   -invalidated-directory=%/t/include/Umb/sub \
// RUN:   -invalidated-directory=%/t/nonexistent 2>&1 | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=FRESH

// FRESH:      "file-deps": [
// FRESH-NEXT:   "[[PREFIX]]/include/Umb/module.modulemap"
// FRESH-NEXT:   "[[PREFIX]]/include/Umb/sub/a.h"
// FRESH-NEXT:   "[[PREFIX]]/include/Umb/sub/b.h"
// FRESH-NEXT: ]
// FRESH:      "name": "UmbDir"

// 5) Paths that differ only in spelling still match: `.` components and a
//    trailing separator are removed before comparison.
// RUN: touch %t/include/Umb/sub/c.h
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full \
// RUN:   -invalidated-directory=%/t/include/./Umb/sub/ 2>&1 | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=SPELLING

// SPELLING:      "file-deps": [
// SPELLING-NEXT:   "[[PREFIX]]/include/Umb/module.modulemap"
// SPELLING-NEXT:   "[[PREFIX]]/include/Umb/sub/a.h"
// SPELLING-NEXT:   "[[PREFIX]]/include/Umb/sub/b.h"
// SPELLING-NEXT:   "[[PREFIX]]/include/Umb/sub/c.h"
// SPELLING-NEXT: ]
// SPELLING:      "name": "UmbDir"

// 6) An inferred framework enumerates its Frameworks subdirectory, since anything
//    dropped in there becomes a submodule.
// RUN: sed "s|DIR|%/t|g" %t/fw-cdb.json.template > %t/fw-cdb.json
// RUN: clang-scan-deps -compilation-database %t/fw-cdb.json \
// RUN:   -format experimental-full 2>&1 | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=FW

// FW:      "directory-deps": [
// FW-NEXT:   "[[PREFIX]]/frameworks/Inferred.framework/Frameworks",
// FW-NEXT:   "[[PREFIX]]/frameworks/Inferred.framework/Headers"
// FW-NEXT: ]
// FW:      "name": "Inferred"

// RUN: mkdir -p %t/frameworks/Inferred.framework/Frameworks/Sub.framework/Headers
// RUN: echo "// sub" > %t/frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h
// RUN: clang-scan-deps -compilation-database %t/fw-cdb.json \
// RUN:   -format experimental-full \
// RUN:   -invalidated-directory=%/t/frameworks/Inferred.framework/Frameworks 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=FW-SUB

// FW-SUB: "[[PREFIX]]/frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h"

//--- frameworks/module.modulemap
framework module * {}

//--- frameworks/Inferred.framework/Headers/Inferred.h

//--- fw.m
#import <Inferred/Inferred.h>

//--- fw-cdb.json.template
[{
  "file": "DIR/fw.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/fwcache -F DIR/frameworks -c DIR/fw.m -o DIR/fw.o"
}]
