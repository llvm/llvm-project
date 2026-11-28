// Diagnostics on a preprocessed input are reported at the location presumed by
// its line markers, not at the physical location in the preprocessed file.

// RUN: clang-tidy %S/Inputs/preprocessed-input/main.ii \
// RUN:   -checks='-*,google-explicit-constructor,readability-inconsistent-declaration-parameter-name' \
// RUN:   -- -std=c++17 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MESSAGES --implicit-check-not='main.ii:'

// The note is located before the warning in the preprocessed file.
// CHECK-MESSAGES: main.cpp:50:6: warning: function 'definedInHeader' has a definition with different parameter names [readability-inconsistent-declaration-parameter-name]
// CHECK-MESSAGES-NEXT: 50 | void definedInHeader(int b);
// CHECK-MESSAGES: some/dir/foo.h:20:6: note: the definition seen here
// CHECK-MESSAGES-NEXT: 20 | void definedInHeader(int a) {}
// CHECK-MESSAGES: main.cpp:50:6: note: differing parameters are named here: ('b'), in definition: ('a')
// CHECK-MESSAGES: main.cpp:52:11: warning: single-argument constructors must be marked explicit
// Make sure diagnostics in system headers are still suppressed (line marker flag 3).
// CHECK-MESSAGES-NOT: system.h
// CHECK-MESSAGES: main.cpp:53:11: warning: single-argument constructors must be marked explicit

// Fixes should be applied to the preprocessed file, and not the fix notes.
// We cannot assume the presumed location is available on the executor.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp %S/Inputs/preprocessed-input/main.ii %t/main.ii
// RUN: clang-tidy %t/main.ii -checks='-*,google-explicit-constructor' --fix \
// RUN:   -- -std=c++17 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-FIX-MESSAGES --implicit-check-not='main.ii:'
// RUN: FileCheck %s --check-prefix=CHECK-FIXES --input-file=%t/main.ii

// CHECK-FIX-MESSAGES: main.cpp:52:11: warning: single-argument constructors must be marked explicit
// CHECK-FIX-MESSAGES: main.cpp:52:11: note: FIX-IT applied suggested code changes
// CHECK-FIX-MESSAGES: main.cpp:53:11: warning: single-argument constructors must be marked explicit
// CHECK-FIX-MESSAGES: main.cpp:53:11: note: FIX-IT applied suggested code changes

// CHECK-FIXES: class A { explicit A(int); };
// CHECK-FIXES: class S { S(int); };
// CHECK-FIXES: class B { explicit B(int); };

// The physical location is reported when presumed locations are disabled.
// RUN: clang-tidy %S/Inputs/preprocessed-input/main.ii \
// RUN:   -checks='-*,google-explicit-constructor' \
// RUN:   -- -std=c++17 -Xclang -fno-diagnostics-use-presumed-location 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PRESUMED --implicit-check-not='main.cpp:'

// CHECK-NO-PRESUMED: main.ii:15:11: warning: single-argument constructors must be marked explicit
// CHECK-NO-PRESUMED: main.ii:19:11: warning: single-argument constructors must be marked explicit

// Compiler diagnostics are emitted while the input is being preprocessed, 
// Diagnostics emitted afterwards must still take the later line markers into account.
// RUN: clang-tidy %S/Inputs/preprocessed-input/compiler-diag.ii \
// RUN:   -checks='-*,clang-diagnostic-return-type,google-explicit-constructor' \
// RUN:   -- -std=c++17 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-COMPILER-DIAG --implicit-check-not='compiler-diag.ii:'

// CHECK-COMPILER-DIAG: other.cpp:10:10: warning: non-void function does not return a value [clang-diagnostic-return-type]
// CHECK-COMPILER-DIAG: other.cpp:20:11: warning: single-argument constructors must be marked explicit
