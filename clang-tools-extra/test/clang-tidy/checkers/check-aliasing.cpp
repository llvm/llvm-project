// Test the clang-tidy check alias framework using permanent cert aliases.
//
// Test: enabling alias enables canonical check.
// RUN: clang-tidy --list-checks --checks='-*,cert-oop11-cpp' 2>&1 | FileCheck -check-prefix=ENABLE %s
//
// Test: disabling canonical disables alias.
// RUN: clang-tidy --list-checks --allow-no-checks --checks='-*,cert-oop11-cpp,-performance-move-constructor-init' 2>&1 | FileCheck -check-prefix=DISABLE %s
//
// Test: disabling alias disables canonical.
// RUN: clang-tidy --list-checks --allow-no-checks --checks='-*,performance-move-constructor-init,-cert-oop11-cpp' 2>&1 | FileCheck -check-prefix=DISABLE-REVERSE %s
//
// Test: --notify-aliases emits note for alias in --checks.
// RUN: clang-tidy %s --notify-aliases --checks='-*,cert-dcl03-c' -- -std=c++11 2>&1 | FileCheck -check-prefix=NOTIFY %s
//
// Test: NOLINT with alias name suppresses canonical diagnostic.
// RUN: clang-tidy %s --checks='-*,cert-oop11-cpp' -- -std=c++11 2>&1 | FileCheck -check-prefix=NOLINT-ALIAS %s
//
// Test: --warnings-as-errors with alias name promotes canonical diagnostic.
// RUN: not clang-tidy %s --checks='-*,performance-move-constructor-init' --warnings-as-errors='cert-oop11-cpp' -- -std=c++11 2>&1 | FileCheck -check-prefix=WARN-AS-ERR %s

// ENABLE-DAG: cert-oop11-cpp
// ENABLE-DAG: performance-move-constructor-init

// DISABLE-NOT: cert-oop11-cpp
// DISABLE-NOT: performance-move-constructor-init

// DISABLE-REVERSE-NOT: cert-oop11-cpp
// DISABLE-REVERSE-NOT: performance-move-constructor-init

// NOTIFY: note: 'cert-dcl03-c' is an alias for canonical name 'misc-static-assert' [checker-alias]

struct B {
  B(B &&) noexcept = default;
  B(const B &) = default;
  B &operator=(const B &) = default;
  ~B() {}
};

struct D {
  B b;
  D(D &&d) : b(d.b) {} // NOLINT(cert-oop11-cpp)
  // NOLINT-ALIAS: Suppressed 1 warnings (1 NOLINT)
};

struct E {
  B b;
  // WARN-AS-ERR: :[[@LINE+1]]:{{.*}} error: {{.*}} [performance-move-constructor-init,-warnings-as-errors]
  E(E &&e) : b(e.b) {}
};
