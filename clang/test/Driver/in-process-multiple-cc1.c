// REQUIRES: llvm-driver, webassembly-registered-target

// A Clang invocation owned by ToolSession can execute multiple cc1 jobs
// in-process. Each job must free its CompilerInstance before returning.
// RUN: split-file %s %t
// RUN: cd %t && %clang -fintegrated-cc1 --target=wasm32-unknown-unknown -c -### \
// RUN:   first.c second.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COMMANDS \
// RUN:       --implicit-check-not='"-disable-free"'
// COMMANDS-COUNT-2: (in-process)

// The same cleanup rule applies to a single cc1 job because the session stays
// alive after the top-level Clang invocation returns.
// RUN: cd %t && %clang -fintegrated-cc1 --target=wasm32-unknown-unknown -c -### first.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SINGLE \
// RUN:       --implicit-check-not='"-disable-free"'
// SINGLE: (in-process)

// An explicit request for a separate cc1 process remains authoritative.
// RUN: cd %t && %clang --target=wasm32-unknown-unknown \
// RUN:   -fno-integrated-cc1 -c -### first.c second.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPAWN \
// RUN:       --implicit-check-not='(in-process)'
// SPAWN-COUNT-2: "-disable-free"

// Exercise the jobs and verify that both objects were emitted successfully.
// RUN: cd %t && %clang -fintegrated-cc1 --target=wasm32-unknown-unknown \
// RUN:   -c first.c second.c
// RUN: llvm-readobj --file-headers %t/first.o %t/second.o \
// RUN:   | FileCheck %s --check-prefix=OBJECTS
// OBJECTS-COUNT-2: Format: WASM

// A failed cc1 job does not prevent the following job from running in the
// same process, regardless of which source comes first.
// RUN: cd %t && not %clang -fintegrated-cc1 --target=wasm32-unknown-unknown \
// RUN:   -c good-before-bad.c bad-after-good.c
// RUN: test -f %t/good-before-bad.o
// RUN: cd %t && not %clang -fintegrated-cc1 --target=wasm32-unknown-unknown \
// RUN:   -c bad-before-good.c good-after-bad.c
// RUN: test -f %t/good-after-bad.o

//--- first.c
int first(void) { return 1; }

//--- second.c
int second(void) { return 2; }

//--- good-before-bad.c
int good_before_bad(void) { return 3; }

//--- bad-after-good.c
int bad_after_good(void) { return does_not_parse( }

//--- bad-before-good.c
int bad_before_good(void) { return does_not_parse( }

//--- good-after-bad.c
int good_after_bad(void) { return 4; }
