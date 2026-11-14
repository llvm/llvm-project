// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s modernize-use-trailing-return-type %t
// RUN: %check_clang_tidy -check-suffixes=,CXX20 -std=c++20 %s modernize-use-trailing-return-type %t
// RUN: %check_clang_tidy -check-suffixes=,CXX20,CXX23 -std=c++23-or-later %s modernize-use-trailing-return-type %t

// The return type has to go after the last token before the body, however that
// token is separated from the body.

#define ID(x) x
#define APPLY_BODY(body) []() body
#define PARENS ()
#define BODY { return 1; }
#define PARENS_AND_BODY () { return 1; }
#define WHOLE_LAMBDA []() { return 1; }

// This has to stay the first macro expansion in the file: the character before
// the body then isn't part of any macro expansion, which used to crash.
auto first_macro_expansion = [] BODY;
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
// CHECK-FIXES: auto first_macro_expansion = [] BODY;

void test_body_in_macro() {
  auto l1 = []() BODY;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = []() BODY;

  auto l2 = [] PARENS_AND_BODY;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l2 = [] PARENS_AND_BODY;

  auto l3 = WHOLE_LAMBDA;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l3 = WHOLE_LAMBDA;

  auto l4 = APPLY_BODY({ return 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l4 = APPLY_BODY({ return 1; });
}

void test_no_space_before_body() {
  int x = 0;

  auto l1 = [](){ return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = []() -> int{ return 1; };

  auto l2 = [](int i){ return i; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l2 = [](int i) -> int{ return i; };

  auto l3 = [x]() mutable{ return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l3 = [x]() mutable -> int{ return x; };

  auto l4 = []() noexcept{ return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l4 = []() noexcept -> int{ return 1; };

  auto l5 = [] PARENS{ return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l5 = [] PARENS -> int{ return 1; };

#if __cplusplus >= 202302L
  auto l6 = []{ return 1; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-CXX23: auto l6 = [] -> int{ return 1; };
#endif
}

void test_comment_before_body() {
  auto l1 = []() /* comment */{ return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = []() -> int /* comment */{ return 1; };

  auto l2 = []() /* comment */ { return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l2 = []() -> int /* comment */ { return 1; };

  auto l3 = []() // comment
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l3 = []() -> int // comment
  { return 1; };
}

void test_lambda_in_macro_argument() {
  auto l1 = ID([]() { return 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = ID([]() -> int { return 1; });

  auto l2 = ID([](){ return 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l2 = ID([]() -> int{ return 1; });

  auto l3 = ID(ID([](){ return 1; }));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l3 = ID(ID([]() -> int{ return 1; }));
}

#if __cplusplus >= 202002L
template <typename T>
void test_requires_clause(T t) {
  auto l1 = [](int i)requires true { return i; };
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-CXX20: auto l1 = [](int i) -> auto requires true { return i; };

  auto l2 = [](int i) /* comment */requires true { return i; };
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-CXX20: auto l2 = [](int i) -> auto /* comment */requires true { return i; };

  auto l3 = [](int i) // comment
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-CXX20: auto l3 = [](int i) -> auto // comment
    requires true { return i; };
}

auto generic_no_space = []<typename T>(T t){ return 1; };
// CHECK-MESSAGES-CXX20: :[[@LINE-1]]:25: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
// CHECK-FIXES-CXX20: auto generic_no_space = []<typename T>(T t) -> auto{ return 1; };
#endif
