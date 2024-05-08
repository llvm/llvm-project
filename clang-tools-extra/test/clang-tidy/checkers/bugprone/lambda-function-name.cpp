// RUN: %check_clang_tidy -check-suffixes=,NO-CONFIG %s bugprone-lambda-function-name %t
// RUN: %check_clang_tidy %s bugprone-lambda-function-name %t -- -config="{CheckOptions: [{key: bugprone-lambda-function-name.IgnoreMacros, value: true}]}" --


void Foo(const char* a, const char* b, int c) {}

#define FUNC_MACRO Foo(__func__, "", 0)
#define FUNCTION_MACRO Foo(__FUNCTION__, "", 0)
#define EMBED_IN_ANOTHER_MACRO1 FUNC_MACRO

void Positives() {
  [] { __func__; }();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: inside a lambda, '__func__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
  [] { __FUNCTION__; }();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: inside a lambda, '__FUNCTION__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
  [] { FUNC_MACRO; }();
  // CHECK-MESSAGES-NO-CONFIG: :[[@LINE-1]]:8: warning: inside a lambda, '__func__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
  [] { FUNCTION_MACRO; }();
  // CHECK-MESSAGES-NO-CONFIG: :[[@LINE-1]]:8: warning: inside a lambda, '__FUNCTION__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
  [] { EMBED_IN_ANOTHER_MACRO1; }();
  // CHECK-MESSAGES-NO-CONFIG: :[[@LINE-1]]:8: warning: inside a lambda, '__func__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
  [] {
    __func__;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inside a lambda, '__func__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
    struct S {
      void f() {
        __func__;
        [] {
          __func__;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inside a lambda, '__func__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
        }();
        __func__;
      }
    };
    __func__;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inside a lambda, '__func__' expands to the name of the function call operator; consider capturing the name of the enclosing function explicitly [bugprone-lambda-function-name]
  }();
}

#define FUNC_MACRO_WITH_FILE_AND_LINE Foo(__func__, __FILE__, __LINE__)
#define FUNCTION_MACRO_WITH_FILE_AND_LINE Foo(__FUNCTION__, __FILE__, __LINE__)
#define EMBED_IN_ANOTHER_MACRO2 FUNC_MACRO_WITH_FILE_AND_LINE

void Negatives() {
  __func__;
  __FUNCTION__;

  // __PRETTY_FUNCTION__ should not trigger a warning because its value is
  // actually potentially useful.
  __PRETTY_FUNCTION__;
  [] { __PRETTY_FUNCTION__; }();

  // Don't warn if __func__/__FUNCTION is used inside a macro that also uses
  // __FILE__ and __LINE__, on the assumption that __FILE__ and __LINE__ will
  // be useful even if __func__/__FUNCTION__ is not.
  [] { FUNC_MACRO_WITH_FILE_AND_LINE; }();
  [] { FUNCTION_MACRO_WITH_FILE_AND_LINE; }();
  [] { EMBED_IN_ANOTHER_MACRO2; }();

  [] (const char* func = __func__) { func; }();
  [func=__func__] { func; }();
}
