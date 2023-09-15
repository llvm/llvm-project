// REQUIRES: shell
// RUN: rm -rf "%t"
// RUN: mkdir "%t"
// RUN: cp "%s" "%t/code.cpp"
// RUN: echo '' > "%t/.clang-tidy"

// Compile commands tests (explicit --driver-mode):
// RUN: echo '[{"directory":"%t/","file":"code.cpp","command":"dummy-compiler /W4 code.cpp"}]' > "%t/compile_commands.json"
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   --extra-arg=--driver-mode=cl 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   --extra-arg-before=--driver-mode=cl 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   --config='{ExtraArgs: ["--driver-mode=cl"]}' 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   --config='{ExtraArgsBefore: ["--driver-mode=cl"]}' 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s

// Compile commands tests (implicit --driver-mode):
// RUN: echo '[{"directory":"%t/","file":"code.cpp","command":"cl.exe /W4 code.cpp"}]' > "%t/compile_commands.json"
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s

// Compile commands tests (negative)
// RUN: echo '[{"directory":"%t/","file":"code.cpp","command":"dummy-compiler -MT /W4 code.cpp"}]' > "%t/compile_commands.json"
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' --check-prefix=NEGATIVE_CHECK --allow-empty %s

// Command line tests (explicit --driver-mode):
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" "%t/code.cpp" \
// RUN:   -- --driver-mode=cl -MD /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" "%t/code.cpp" \
// RUN:   --extra-arg=--driver-mode=cl -- -MD /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" "%t/code.cpp" \
// RUN:   --extra-arg-before=--driver-mode=cl -- -MD /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   --config='{ExtraArgs: ["--driver-mode=cl"]}' -- -MD /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   --config='{ExtraArgsBefore: ["--driver-mode=cl"]}' -- -MD /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s

// Command line tests (implicit --driver-mode):
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   -- cl.exe -MD /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s

// Command line tests (negative)
// RUN: clang-tidy --checks="-*,clang-diagnostic-reorder-ctor,readability-redundant-string-cstr" -p="%t" "%t/code.cpp" \
// RUN:   -- dummy-compiler -MT /W4 code.cpp 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' --check-prefix=NEGATIVE_CHECK --allow-empty %s

struct string {};

struct A {
    A(string aa, string bb) : b(bb), a(aa) {}
// CHECK: code.cpp:[[@LINE-1]]:31: warning: field 'b' will be initialized after field 'a' [clang-diagnostic-reorder-ctor]
// NEGATIVE_CHECK-NOT: warning: field 'b' will be initialized after field 'a' [clang-diagnostic-reorder-ctor]

    string a;
    string b;
};
