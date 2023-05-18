// RUN: %check_clang_tidy -std=c++14 %s bugprone-signal-handler %t -- -- -isystem %clang_tidy_headers -isystem %S/Inputs/signal-handler -target x86_64-unknown-unknown
// FIXME: Fix the checker to work in C++17 or later mode.
#include "stdcpp.h"
#include "stdio.h"

// Functions called "signal" that are different from the system version.
typedef void (*callback_t)(int);
void signal(int, callback_t, int);
namespace ns {
void signal(int, callback_t);
}

extern "C" void handler_unsafe(int) {
  printf("xxx");
}

extern "C" void handler_unsafe_1(int) {
  printf("xxx");
}

namespace test_invalid_handler {

void handler_non_extern_c(int) {
  printf("xxx");
}

struct A {
  static void handler_member(int) {
    printf("xxx");
  }
};

void test() {
  std::signal(SIGINT, handler_unsafe_1);
  // CHECK-MESSAGES: :[[@LINE-17]]:3: warning: standard function 'printf' may not be asynchronous-safe; calling it from a signal handler may be dangerous [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:23: note: function 'handler_unsafe_1' registered here as signal handler

  std::signal(SIGINT, handler_non_extern_c);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: functions without C linkage are not allowed as signal handler (until C++17) [bugprone-signal-handler]
  std::signal(SIGINT, A::handler_member);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: functions without C linkage are not allowed as signal handler (until C++17) [bugprone-signal-handler]
  std::signal(SIGINT, [](int) { printf("xxx"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: lambda function is not allowed as signal handler (until C++17) [bugprone-signal-handler]

  // This case is (deliberately) not found by the checker.
  std::signal(SIGINT, [](int) -> callback_t { return &handler_unsafe; }(1));
}

} // namespace test_invalid_handler

namespace test_non_standard_signal_call {

struct Signal {
  static void signal(int, callback_t);
};

void test() {
  // No diagnostics here. All these signal calls differ from the standard system one.
  signal(SIGINT, handler_unsafe, 1);
  ns::signal(SIGINT, handler_unsafe);
  Signal::signal(SIGINT, handler_unsafe);
  system_other::signal(SIGINT, handler_unsafe);
}

} // namespace test_non_standard_signal_call

namespace test_cpp_construct_in_handler {

struct Struct {
  virtual ~Struct() {}
  void f1();
  int *begin();
  int *end();
  static void f2();
};
struct Derived : public Struct {
};

struct X {
  X(int, float);
};

Struct *S_Global;
const Struct *S_GlobalConst;

void f_non_extern_c() {
}

void f_default_arg(int P1 = 0) {
}

extern "C" void handler_cpp(int) {
  using namespace ::test_cpp_construct_in_handler;

  // These calls are not found as problems.
  // (Called functions are not analyzed if the current function has already
  // other problems.)
  f_non_extern_c();
  Struct::f2();
  // 'auto' is not disallowed
  auto Auto = 28u;

  Struct S;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:10: remark: internally, the statement is parsed as a 'CXXConstructExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  S_Global->f1();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXMemberCallExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  const Struct &SRef = Struct();
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:24: remark: internally, the statement is parsed as a 'CXXBindTemporaryExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  X(3, 4.4);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXTemporaryObjectExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler

  auto L = [](int i) { printf("%d", i); };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:12: remark: internally, the statement is parsed as a 'CXXConstructExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  L(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXOperatorCallExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler

  try {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
    // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXTryStmt'
    // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
    int A;
  } catch (int) {
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-3]]:5: remark: internally, the statement is parsed as a 'CXXCatchStmt'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler

  throw(12);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXThrowExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler

  for (int I : S) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-3]]:3: remark: internally, the statement is parsed as a 'CXXForRangeStmt'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  // CHECK-MESSAGES: :[[@LINE-5]]:14: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-6]]:14: remark: internally, the statement is parsed as a 'CXXMemberCallExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler

  int Int = *(reinterpret_cast<int *>(&S));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:15: remark: internally, the statement is parsed as a 'CXXReinterpretCastExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  Int = static_cast<int>(12.34);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:9: remark: internally, the statement is parsed as a 'CXXStaticCastExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  Derived *Der = dynamic_cast<Derived *>(S_Global);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:18: remark: internally, the statement is parsed as a 'CXXDynamicCastExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  Struct *SPtr = const_cast<Struct *>(S_GlobalConst);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:18: remark: internally, the statement is parsed as a 'CXXConstCastExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  Int = int(12.34);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:9: remark: internally, the statement is parsed as a 'CXXFunctionalCastExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler

  int *IPtr = new int[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:15: remark: internally, the statement is parsed as a 'CXXNewExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  delete[] IPtr;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXDeleteExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  IPtr = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:10: remark: internally, the statement is parsed as a 'CXXNullPtrLiteralExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  bool Bool = true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:15: remark: internally, the statement is parsed as a 'CXXBoolLiteralExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
  f_default_arg();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: remark: internally, the statement is parsed as a 'CXXDefaultArgExpr'
  // CHECK-MESSAGES: :198:23: note: function 'handler_cpp' registered here as signal handler
}

void test() {
  std::signal(SIGINT, handler_cpp);
}

} // namespace test_cpp_construct_in_handler

namespace test_cpp_indirect {

void non_extern_c() {
  int *P = nullptr;
}

extern "C" void call_cpp_indirect() {
  int *P = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: C++-only construct is not allowed in signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE-2]]:12: remark: internally, the statement is parsed as a 'CXXNullPtrLiteralExpr'
  // CHECK-MESSAGES: :[[@LINE+8]]:3: note: function 'call_cpp_indirect' called here from 'handler_cpp_indirect'
  // CHECK-MESSAGES: :[[@LINE+11]]:23: note: function 'handler_cpp_indirect' registered here as signal handler
}

extern "C" void handler_cpp_indirect(int) {
  non_extern_c();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: functions without C linkage are not allowed as signal handler (until C++17) [bugprone-signal-handler]
  // CHECK-MESSAGES: :[[@LINE+5]]:23: note: function 'handler_cpp_indirect' registered here as signal handler
  call_cpp_indirect();
}

void test() {
  std::signal(SIGINT, handler_cpp_indirect);
}

} // namespace test_cpp_indirect
