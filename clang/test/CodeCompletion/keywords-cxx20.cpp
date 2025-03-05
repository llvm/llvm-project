module;

export module M;

export const char8_t x = 1;

template<typename T> requires true
const int y = requires { typename T::type; requires T::value; };

class co_test {};

int f(){ co_test test; return 1; }

module: private;

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:1:3 %s | FileCheck --check-prefix=CHECK-MODULE1 %s
// CHECK-MODULE1: module;
// CHECK-MODULE1: module <#name#>;

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:3:11 %s | FileCheck --check-prefix=CHECK-MODULE2 %s
// CHECK-MODULE2: module <#name#>;

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:14:3 %s | FileCheck --check-prefix=CHECK-MODULE3 %s
// CHECK-MODULE3: module: private;

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:3:3 %s | FileCheck --check-prefix=CHECK-EXPORT %s
// CHECK-EXPORT: export

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:5:11 %s | FileCheck --check-prefix=CHECK-CONST %s
// CHECK-CONST: const
// CHECK-CONST: consteval
// CHECK-CONST: constexpr
// CHECK-CONST: constinit

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:5:19 %s | FileCheck --check-prefix=CHECK-CHAR %s
// CHECK-CHAR: char8_t

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:8:3 %s | FileCheck --check-prefix=CHECK-CONSTRAINT %s
// CHECK-CONSTRAINT: concept
// CHECK-CONSTRAINT: const
// CHECK-CONSTRAINT: consteval
// CHECK-CONSTRAINT: constexpr
// CHECK-CONSTRAINT: constinit

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:7:27 %s | FileCheck --check-prefix=CHECK-REQUIRES2 %s
// CHECK-REQUIRES2: requires

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:8:20 %s | FileCheck -check-prefix=CHECK-REQUIRE %s
// CHECK-REQUIRE: [#bool#]requires (<#parameters#>) {
// CHECK-REQUIRE: <#requirements#>
// CHECK-REQUIRE: }

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:12:13 %s | FileCheck --check-prefix=CHECK-COROUTINE %s
// CHECK-COROUTINE: co_await <#expression#>
// CHECK-COROUTINE: co_return <#expression#>;
// CHECK-COROUTINE: co_yield <#expression#>

