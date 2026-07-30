int function(int x) {
  return x + 1;
}

int variable = 0;

class Class {
public:
  Class() { }

  int method(int x) {
    return x + 1;
  }

  virtual void virtualMethod() {
  }

  static void staticMethod() {
  }

  static int staticVar;
};

class SubClass : public Class {
  void virtualMethod() override final {
  }
};

struct Struct {
};

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:1:1 %s | FileCheck --check-prefix=CHECK-TOP-LEVEL %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:5:1 %s | FileCheck --check-prefix=CHECK-TOP-LEVEL %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:11:1 %s | FileCheck --check-prefix=CHECK-TOP-LEVEL %s
// CHECK-TOP-LEVEL: alignas(<#expression#>)
// CHECK-TOP-LEVEL: constexpr
// CHECK-TOP-LEVEL: static_assert(<#expression#>, <#message#>);
// CHECK-TOP-LEVEL: thread_local
// CHECK-TOP-LEVEL-NOT: final
// CHECK-TOP-LEVEL-NOT: noexcept

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:1:14 %s | FileCheck --check-prefix=CHECK-PARAM %s
// CHECK-PARAM-NOT: alignas
// CHECK-PARAM-NOT: constexpr
// CHECK-PARAM-NOT: final
// CHECK-PARAM-NOT: thread_local

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:21:10 %s | FileCheck --check-prefix=CHECK-STATICVAR1 %s
// CHECK-STATICVAR1: constexpr
// CHECK-STATICVAR1: thread_local

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:7:13 %s | FileCheck --check-prefix=CHECK-CLASS-QUALIFIER %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:24:16 %s | FileCheck --check-prefix=CHECK-CLASS-QUALIFIER %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:29:15 %s | FileCheck --check-prefix=CHECK-CLASS-QUALIFIER %s
// CHECK-CLASS-QUALIFIER: final

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:1:21 %s | FileCheck --check-prefix=CHECK-FUNCTION-QUALIFIER %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:9:11 %s | FileCheck --check-prefix=CHECK-FUNCTION-QUALIFIER %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:18:30 %s | FileCheck --check-prefix=CHECK-FUNCTION-QUALIFIER %s
// CHECK-FUNCTION-QUALIFIER: noexcept
// CHECK-FUNCTION-QUALIFIER-NOT: final
// CHECK-FUNCTION-QUALIFIER-NOT: override

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:11:21 %s | FileCheck --check-prefix=CHECK-METHOD-QUALIFIER %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:15:32 %s | FileCheck --check-prefix=CHECK-METHOD-QUALIFIER %s
// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:25:24 %s | FileCheck --check-prefix=CHECK-METHOD-QUALIFIER %s
// CHECK-METHOD-QUALIFIER: final
// CHECK-METHOD-QUALIFIER: noexcept
// CHECK-METHOD-QUALIFIER: override

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:25:33 %s | FileCheck --check-prefix=CHECK-OVERRIDE-SPECIFIED %s
// CHECK-OVERRIDE-SPECIFIED: final
// CHECK-OVERRIDE-SPECIFIED: noexcept
// CHECK-OVERRIDE-SPECIFIED-NOT: override

// RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:25:39 %s | FileCheck --check-prefix=CHECK-OVERRIDE-FINAL-SPECIFIED %s
// CHECK-OVERRIDE-FINAL-SPECIFIED: noexcept
// CHECK-OVERRIDE-FINAL-SPECIFIED-NOT: final
// CHECK-OVERRIDE-FINAL-SPECIFIED-NOT: override
