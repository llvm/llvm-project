// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Systematic tests for [[clang::suppress]] interaction with lambdas.

// Placeholder type for triggering instantiations.
struct Type{};

// ============================================================================
// Group A: Lambda in suppressed statement block
// ============================================================================

void lambda_in_suppressed_block() {
  [[clang::suppress]] {
    auto lam = []() {
      clang_analyzer_warnIfReached(); // no-warning
    };
    lam();
  }
}

void lambda_in_unsuppressed_block() {
  auto lam = []() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  };
  lam();
}

// ============================================================================
// Group B: Lambda in suppressed class method
// ============================================================================

struct [[clang::suppress]] B_SuppressedClass {
  void method_with_lambda() {
    auto lam = []() {
      clang_analyzer_warnIfReached(); // no-warning
    };
    lam();
  }
};

struct B_UnsuppressedClass {
  void method_with_lambda() {
    auto lam = []() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    };
    lam();
  }
};

void test_B() {
  B_SuppressedClass().method_with_lambda();
  B_UnsuppressedClass().method_with_lambda();
}

// ============================================================================
// Group C: Nested lambdas
// ============================================================================

void nested_lambda_suppressed() {
  [[clang::suppress]] {
    auto outer = []() {
      auto inner = []() {
        clang_analyzer_warnIfReached(); // no-warning
      };
      return inner();
    };
    return outer(); // no-warning
  }
}

void nested_lambda_unsuppressed() {
  auto outer = []() {
    auto inner = []() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    };
    inner();
  };
  outer();
}

// ============================================================================
// Group D: Lambda with captures
// ============================================================================

int lambda_with_ref_capture_suppressed() {
  int *x = 0;
  [[clang::suppress]] {
    auto lam = [&x]() {
      return *x;
    };
    return lam(); // no-warning
  }
}

int lambda_with_ref_capture_unsuppressed() {
  int *x = 0;
  auto lam = [&x]() {
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  };
  return lam();
}

int lambda_capture_by_value_suppressed() {
  int *x = 0;
  [[clang::suppress]] {
    auto lam = [x]() {
      return *x;
    };
    return lam(); // no-warning
  }
}

// ============================================================================
// Group E: Lambda in suppressed namespace
// ============================================================================

namespace [[clang::suppress]] SuppressedNS {
  void func_with_lambda() {
    auto lam = []() {
      clang_analyzer_warnIfReached(); // no-warning
    };
    lam();
  }
} // namespace SuppressedNS

// ============================================================================
// Group F: Suppressed lambda, unsuppressed enclosing
// ============================================================================

void selective_suppression_unsup() {
  auto unsuppressed_lam = []() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  };
  unsuppressed_lam();
}

void selective_suppression_sup() {
  [[clang::suppress]] auto suppressed_lam = []() {
    clang_analyzer_warnIfReached(); // no-warning
  };
  suppressed_lam();
}

// ============================================================================
// Group G: Lambda in template function
// ============================================================================

template <typename T>
[[clang::suppress]] void tmpl_func_with_lambda(T) {
  auto lam = []() {
    clang_analyzer_warnIfReached(); // no-warning
  };
  lam();
}

template <typename T>
void tmpl_func_with_lambda_unsuppressed(T) {
  auto lam = []() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  };
  lam();
}

void test_G() {
  tmpl_func_with_lambda(Type{});
  tmpl_func_with_lambda_unsuppressed(Type{});
}

// ============================================================================
// Group H: Lambda in suppressed class template
// ============================================================================

template <typename T>
struct [[clang::suppress]] H_SuppressedTmpl {
  void method() {
    auto lam = []() {
      clang_analyzer_warnIfReached(); // no-warning
    };
    lam();
  }
};

template <typename T>
struct H_UnsuppressedTmpl {
  void method() {
    auto lam = []() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    };
    lam();
  }
};

void test_H() {
  H_SuppressedTmpl<Type>().method();
  H_UnsuppressedTmpl<Type>().method();
}

// ============================================================================
// Group I: Immediately-invoked lambda expression
// ============================================================================

int iile_suppressed() {
  [[clang::suppress]] {
    return []() {
      int *x = 0;
      return *x;
    }(); // no-warning
  }
}

int iile_unsuppressed() {
  return []() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }();
}

// ============================================================================
// Group J: Generic lambda (C++14)
// ============================================================================

void generic_lambda_suppressed() {
  [[clang::suppress]] {
    auto lam = [](auto) {
      clang_analyzer_warnIfReached(); // no-warning
    };
    lam(Type{});
  }
}

void generic_lambda_unsuppressed() {
  auto lam = [](auto) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  };
  lam(Type{});
}
