const char8_t x = 1;

template<typename T> requires true
const int y = requires { typename T::type; requires T::value; };

int f(){ co_await 1; }

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:1:3 %s | FileCheck --check-prefix=CHECK-TOP-LEVEL %s
// CHECK-TOP-LEVEL: const
// CHECK-TOP-LEVEL: consteval
// CHECK-TOP-LEVEL: constexpr
// CHECK-TOP-LEVEL: constinit

// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:1:12 %s | FileCheck --check-prefix=CHECK-TOP-LEVEL %s
// CHECK-TOP-LEVEL: char8_t

// RUN: %clang-cc1 -std=c++20 -code-completion-at=%s:4:3 %s | FileCheck --check-prefix=CHECK-REQUIRES %s
// CHECK-REQUIRES: concept
// CHECK-REQUIRES: const
// CHECK-REQUIRES: consteval
// CHECK-REQUIRES: constexpr
// CHECK-REQUIRES: constinit

// RUN: %clang-cc1 -std=c++20 -code-completion-at=%s:3:27 %s | FileCheck --check-prefix=CHECK-REQUIRES %s
// CHECK-REQUIRES: requires

// RUN: %clang-cc1 -std=c++20 -code-completion-at=%s:4:20 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1-NEXT: COMPLETION: Pattern: [#bool#]requires (<#parameters#>) {
// CHECK-CC1-NEXT: <#requirements#>
// CHECK-CC1-NEXT: }

// RUN: %clang-cc1 -std=c++20 -code-completion-at=%s:6:13 %s | FileCheck --check-prefix=CHECK-COAWAIT %s
// CHECK-COAWAIT: Pattern : co_await <#expression#>
// CHECK-COAWAIT: Pattern : co_return <#expression#>;
// CHECK-COAWAIT: Pattern : co_yield <#expression#>