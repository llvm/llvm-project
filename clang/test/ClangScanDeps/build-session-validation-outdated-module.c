// Test build session validation with parallel compilation for incremental builds.

// RUN: rm -rf %t
// RUN: split-file %s %t
// Use the same command line arguments to ensure .pcm files are in the same location.
// RUN: echo "-fsyntax-only -I %/t/include"                      > %t/ctx.rsp
// RUN: echo "-fmodules -fmodules-cache-path=%/t/module-cache"  >> %t/ctx.rsp
// RUN: echo "-fbuild-session-file=%/t/session.timestamp"       >> %t/ctx.rsp
// RUN: echo "-fmodules-validate-once-per-build-session"        >> %t/ctx.rsp
//
// Add a build session file to avoid errors that it's missing. Doesn't do anything as we start with a clean build.
// RUN: touch %t/session.timestamp
// Generate headers that take noticeable time to parse. The purpose is to tweak the timing, so can reproduce the error.
// RUN: %python %t/generate-expensive-header.py A 2000000 > %t/include/expensive-header.h
// RUN: %python %t/generate-expensive-header.py B 500000 > %t/include/less-expensive-header.h
//
// Create SignatureExpectation.pcm with a specific Target signature.
// RUN: cp %t/version1.h %t/include/target.h
// RUN: sed "s|DIR|%/t|g" %t/prepare-expected-signature.json.template > %t/prepare-expected-signature.json
// RUN: clang-scan-deps -compilation-database %t/prepare-expected-signature.json > /dev/null
//
// Built a different Target.pcm with a different signature.
// RUN: cp %t/version2.h %t/include/target.h
// Update the build session to rebuild Target.
// RUN: touch %t/session.timestamp
// RUN: sed "s|DIR|%/t|g" %t/create-outdated-module.json.template > %t/create-outdated-module.json
// RUN: clang-scan-deps -compilation-database %t/create-outdated-module.json > /dev/null
//
// Simulate the incremental build.
// RUN: cp %t/version3.h %t/include/target.h
// RUN: touch %t/session.timestamp
// Scan 2 files in parallel.
// RUN: sed "s|DIR|%/t|g" %t/incremental.json.template > %t/incremental.json
// RUN: clang-scan-deps -compilation-database %t/incremental.json > /dev/null

//--- include/module.modulemap
module Target {
  header "target.h"
  export *
}

module SignatureExpectation {
  header "signature-expectation.h"
  textual header "expensive-header.h"
  export *
}

//--- version1.h
// empty

//--- version2.h
void some_func(void);

//--- version3.h
void some_func(void);
#define TARGET_MACRO 1

//--- generate-expensive-header.py
import sys

num_items = int(sys.argv[2])
suffix = sys.argv[1]
for i in range(num_items):
    if i == 0:
        print(f"#define MACRO_{suffix}_{i} 0")
    else:
        print(f"#define MACRO_{suffix}_{i} MACRO_{suffix}_{i-1} + 1")
    
//--- include/signature-expectation.h
// Expensive header is used to slow down the build of SignatureExpectation module.
// Without the slow down we are able to validate and rebuild Target before another thread can do it.
#include <expensive-header.h>
#include <target.h>

//--- prepare-expected-signature.c
#include <signature-expectation.h>
//--- prepare-expected-signature.json.template
[{                                                                              
  "file": "DIR/prepare-expected-signature.c",
  "directory": "DIR",
  "command": "clang @DIR/ctx.rsp DIR/prepare-expected-signature.c"
}] 

//--- create-outdated-module.c
#include <target.h>
//--- create-outdated-module.json.template
[{                                                                              
  "file": "DIR/create-outdated-module.c",
  "directory": "DIR",
  "command": "clang @DIR/ctx.rsp DIR/create-outdated-module.c"
}] 

//--- use-outdated-module.c
// At first load SignatureExpectation module, so it would load Target but
// reject it due to a signature mismatch.
// After this we rebuild SignatureExpectation and try to use already loaded
// Target buffer when encounter corresponding include in the source code.
// The problem is that it corresponds to version2.h and not version3.h.
#include <signature-expectation.h>

#ifndef TARGET_MACRO
# error Macro is missing, probably using outdated Target module
# include <missing macro>
#endif

//--- rebuild-outdated-module.c
// We aim for another thread to load the outdated module Target with
// the wrong signature before we rebuild it.
#include <less-expensive-header.h>
#include <target.h>

//--- incremental.json.template
[{                                                                              
  "file": "DIR/use-outdated-module.c",
  "directory": "DIR",
  "command": "clang @DIR/ctx.rsp DIR/use-outdated-module.c"
},
{                                                                              
  "file": "DIR/rebuild-outdated-module.c",
  "directory": "DIR",
  "command": "clang @DIR/ctx.rsp DIR/rebuild-outdated-module.c"
}] 
