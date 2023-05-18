// RUN: %clang_analyze_cc1 %s -verify=fn-pointer \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=true \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=false \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=false

// RUN: %clang_analyze_cc1 %s -verify=param-count \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=false \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=true \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=false \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=false

// RUN: %clang_analyze_cc1 %s -verify=method \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=false \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=true \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=false \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=false

// RUN: %clang_analyze_cc1 %s -verify=delete \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=false \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=false \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=false

// RUN: %clang_analyze_cc1 %s -verify=arg-init \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=false \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=false \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=false

// Testing for ArgPointeeInitializedness is in call-and-message.c.

// RUN: %clang_analyze_cc1 %s \
// RUN:   -verify=fn-pointer,param-count,method,delete,arg-init \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | FileCheck %s

namespace function_pointer {
using Fn = void (*)();

void uninit() {
  Fn f;
  f(); // fn-pointer-warning{{Called function pointer is an uninitialized pointer value [core.CallAndMessage]}}
}

void null() {
  Fn f = nullptr;
  f(); // fn-pointer-warning{{Called function pointer is null (null dereference) [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:FunctionPointer from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>eb2083c01775eef452afa75728dd4d8f</string>
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>407c50d9bedd8db28bf34f9411308100</string>

} // namespace function_pointer

namespace wrong_param_count {
using FnOneParam = void (*)(int);
using FnTwoParam = void (*)(int, int);

void f(int, int) {}

void wrong_cast() {
  FnTwoParam f1 = f;
  FnOneParam f2 = reinterpret_cast<FnOneParam>(f1);
  f2(5); // param-count-warning{{Function taking 2 arguments is called with fewer (1) [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:ParameterCount from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>9ff0e9b728422017945c9d5a673de223</string>
} // namespace wrong_param_count

namespace method_call {
struct A {
  void m();
};

void uninit() {
  A *a;
  a->m(); // method-warning{{Called C++ object pointer is uninitialized [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:CXXThisMethodCall from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>7bc35c70465837948a3f5018f27b21cd</string>

void null() {
  A *a = nullptr;
  a->m(); // method-warning{{Called C++ object pointer is null [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:CXXThisMethodCall from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>8ec260c9ef11d7c51fa872212df1163f</string>
} // namespace method_call

namespace operator_delete {
void f() {
  int *i;
  delete i; // delete-warning{{Argument to 'delete' is uninitialized [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:CXXDeallocationArg from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>a8ff99ebaa8746457d3e14af8ef7e75c</string>
} // namespace operator_delete

namespace uninit_arg {
template <class T>
void consume(T);

void fundamental_uninit() {
  int i;
  consume(i); // arg-init-warning{{1st function call argument is an uninitialized value [core.CallAndMessage]}}
}

struct A {
  int i;
};

void record_uninit() {
  A a;
  consume(a); // arg-init-warning{{Passed-by-value struct argument contains uninitialized data (e.g., field: 'i') [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:ArgInitializedness from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>a46bb5c1ee44d4611ffeb13f7f499605</string>
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>e0e0d30ea5a7b2e3a71e1931fa0768a5</string>
} // namespace uninit_arg
