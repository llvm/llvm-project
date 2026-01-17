// Test build session validation with parallel compilation for incremental builds.
//
// Shell is required for clang execution in background.
// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t
// Use the same command line arguments to ensure .pcm files are in the same location.
// RUN: echo "-fsyntax-only -I %/t/include"                      > %t/ctx.rsp
// RUN: echo "-fmodules -fmodules-cache-path=%/t/module-cache"  >> %t/ctx.rsp
// RUN: echo "-fbuild-session-file=%/t/session.timestamp"       >> %t/ctx.rsp
// RUN: echo "-fmodules-validate-once-per-build-session"        >> %t/ctx.rsp
// RUN: echo "-Xclang -fno-modules-force-validate-user-headers" >> %t/ctx.rsp
//
// Add a build session file to avoid errors that it's missing. Doesn't do anything as we start with a clean build.
// RUN: touch %t/session.timestamp
// Generate a header that takes noticeable time to parse.
// RUN: %python %t/generate-expensive-header.py > %t/include/expensive-header.h
//
// Create SignatureExpectation.pcm with a specific Target signature.
// RUN: cp %t/version1.h %t/include/target.h
// RUN: echo "#include <signature-expectation.h>" | %clang @%t/ctx.rsp -x c -
//
// Built a different Target.pcm with a different signature.
// RUN: cp %t/version2.h %t/include/target.h
// Update the build session to rebuild Target.
// RUN: touch %t/session.timestamp
// RUN: echo "#include <target.h>" | %clang @%t/ctx.rsp -x c -
//
// Simulate the incremental build.
// RUN: cp %t/version3.h %t/include/target.h
// RUN: touch %t/session.timestamp
// RUN: sleep 1s
// Compile 2 files in parallel.
// RUN: %clang @%t/ctx.rsp %t/use-outdated-module.c &
// RUN: %clang @%t/ctx.rsp %t/rebuild-outdated-module.c
// RUN: wait

//--- include/module.modulemap
module Target {
  header "target.h"
  textual header "expensive-header.h"
  export *
}

module SignatureExpectation {
  header "signature-expectation.h"
  export *
}

//--- version1.h
// empty

//--- version2.h
void some_func(void);

//--- version3.h
#define TARGET_MACRO 1

//--- generate-expensive-header.py
num_items = 500000
for i in range(num_items):
    print(f"struct struct_{i} {{ int x; float *y; }};")
    print(f"void func_{i}(int *x);")
    
//--- include/signature-expectation.h
// Expensive header is used to slow down the build of SignatureExpectation module.
// Without the slow down we are able to validate and rebuild Target before another process can do it.
#include <expensive-header.h>
#include <target.h>

//--- use-outdated-module.c
// At first load SignatureExpectation module, so it would load Target but
// reject it due to a signature mismatch.
// After this we rebuild SignatureExpectation and try to use already loaded
// Target buffer when encounter corresponding include in the source code.
#include <signature-expectation.h>

#ifndef TARGET_MACRO
# error Macro is missing, probably using outdated Target module
#endif

//--- rebuild-outdated-module.c
// We aim for another process to load the outdated module Target with
// the wrong signature before we rebuild it.
#include <target.h>
