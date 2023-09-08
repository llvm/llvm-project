// RUN: %check_clang_tidy %s llvmlibc-inline-function-decl %t

#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_LLVMLIBC_INLINEFUNCTIONDECL_H
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_LLVMLIBC_INLINEFUNCTIONDECL_H

#define LIBC_INLINE inline

namespace __llvm_libc {

LIBC_INLINE int addi(int a, int b) {
  return a + b;
}

LIBC_INLINE constexpr long addl(long a, long b) {
  return a + b;
}

constexpr long long addll(long long a, long long b) {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'addll' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: LIBC_INLINE constexpr long long addll(long long a, long long b) {
  return a + b;
}

inline unsigned long addul(unsigned long a, unsigned long b) {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'addul' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: LIBC_INLINE inline unsigned long addul(unsigned long a, unsigned long b) {
  return a + b;
}

class  MyClass {
  int A;
public:
  MyClass() : A(123) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'MyClass' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
  // CHECK-FIXES:   LIBC_INLINE MyClass() : A(123) {}

  LIBC_INLINE MyClass(int V) : A(V) {}

  constexpr operator int() const { return A; }
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator int' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
  // CHECK-FIXES:   LIBC_INLINE constexpr operator int() const { return A; }

  LIBC_INLINE bool operator==(const MyClass &RHS) {
    return RHS.A == A;
  }

  static int getVal(const MyClass &V) {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'getVal' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
  // CHECK-FIXES:   LIBC_INLINE static int getVal(const MyClass &V) {
    return V.A;
  }

  LIBC_INLINE static void setVal(MyClass &V, int A) {
    V.A = A;
  }

  constexpr static int addInt(MyClass &V, int A) {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'addInt' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
  // CHECK-FIXES:   LIBC_INLINE constexpr static int addInt(MyClass &V, int A) {
    return V.A += A;
  }

  static LIBC_INLINE int mulInt(MyClass &V, int A) {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'mulInt' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
    return V.A *= A;
  }
};

LIBC_INLINE void lambda() {
// CHECK-MESSAGES-NOT: :[[@LINE+4]]:3: warning: '__invoke' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-MESSAGES-NOT: :[[@LINE+3]]:3: warning: 'operator void (*)()' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-MESSAGES-NOT: :[[@LINE+2]]:3: warning: '~(lambda at [[FILENAME:.+]])' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-MESSAGES-NOT: :[[@LINE+1]]:6: warning: 'operator()' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
  [](){};
}

namespace issue_62746 {

void goodSimpleFunction();
void badSimpleFunction();
void badSimpleFunctionWrongLocation();

LIBC_INLINE void goodSimpleFunction() {}

inline void badSimpleFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'badSimpleFunction' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: LIBC_INLINE inline void badSimpleFunction() {}

void LIBC_INLINE badSimpleFunctionWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'badSimpleFunctionWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename T>
void goodTemplateFunction();
template <typename T>
void badTemplateFunction();
template <typename T>
void badTemplateFunctionWrongLocation();

template <typename T> LIBC_INLINE void goodTemplateFunction() {}

template <typename T> inline void badTemplateFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'badTemplateFunction' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename T> LIBC_INLINE inline void badTemplateFunction() {}

template <typename T> void LIBC_INLINE badTemplateFunctionWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'badTemplateFunctionWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename... Ts>
void goodVariadicFunction();
template <typename... Ts>
void badVariadicFunction();
template <typename... Ts>
void badVariadicFunctionWrongLocation();

template <typename... Ts> LIBC_INLINE void goodVariadicFunction() {}

template <typename... Ts> inline void badVariadicFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'badVariadicFunction' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename... Ts> LIBC_INLINE inline void badVariadicFunction() {}

template <typename... Ts> void LIBC_INLINE badVariadicFunctionWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'badVariadicFunctionWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename... Ts> LIBC_INLINE void LIBC_INLINE badVariadicFunctionWrongLocation() {}

struct NoTemplate {
  void goodNoTemplate();
  void badNoTemplate();
  void badNoTemplateWrongLocation();

  template <typename T>
  void goodNestedTemplate();
  template <typename T>
  void badNestedTemplate();
  template <typename T>
  void badNestedTemplateWrongLocation();

  template <typename... Ts>
  void goodVariadicTemplate();
  template <typename... Ts>
  void badVariadicTemplate();
  template <typename... Ts>
  void badVariadicTemplateWrongLocation();

};

LIBC_INLINE void NoTemplate::goodNoTemplate() {}

inline void NoTemplate::badNoTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'badNoTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: LIBC_INLINE inline void NoTemplate::badNoTemplate() {}

void LIBC_INLINE NoTemplate::badNoTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'badNoTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: LIBC_INLINE void LIBC_INLINE NoTemplate::badNoTemplateWrongLocation() {}

template <typename T> LIBC_INLINE void NoTemplate::goodNestedTemplate() {}

template <typename T> inline void NoTemplate::badNestedTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'badNestedTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename T> LIBC_INLINE inline void NoTemplate::badNestedTemplate() {}

template <typename T> void LIBC_INLINE NoTemplate::badNestedTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'badNestedTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename T> LIBC_INLINE void LIBC_INLINE NoTemplate::badNestedTemplateWrongLocation() {}

template <typename... Ts> LIBC_INLINE void NoTemplate::goodVariadicTemplate() {}

template <typename... Ts> void inline NoTemplate::badVariadicTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'badVariadicTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename... Ts> LIBC_INLINE void inline NoTemplate::badVariadicTemplate() {}

template <typename... Ts> void LIBC_INLINE NoTemplate::badVariadicTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'badVariadicTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename T>
struct SimpleTemplate {
  void goodSimpleTemplate();
  void badSimpleTemplate();
  void badSimpleTemplateWrongLocation();

  template <typename U>
  void goodNestedTemplate();
  template <typename U>
  void badNestedTemplate();
  template <typename U>
  void badNestedTemplateWrongLocation();

  template <typename... Ts>
  void goodNestedVariadicTemplate();
  template <typename... Ts>
  void badNestedVariadicTemplate();
  template <typename... Ts>
  void badNestedVariadicTemplateWrongLocation();
};

template <typename T> LIBC_INLINE void SimpleTemplate<T>::goodSimpleTemplate() {}

template <typename T> inline void SimpleTemplate<T>::badSimpleTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'badSimpleTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename T> LIBC_INLINE inline void SimpleTemplate<T>::badSimpleTemplate() {}

template <typename T> void LIBC_INLINE SimpleTemplate<T>::badSimpleTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'badSimpleTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename T> template <typename U> LIBC_INLINE void SimpleTemplate<T>::goodNestedTemplate() {}

template <typename T> template <typename U> inline void SimpleTemplate<T>::badNestedTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: 'badNestedTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename T> template <typename U> LIBC_INLINE inline void SimpleTemplate<T>::badNestedTemplate() {}

template <typename T> template <typename U> void LIBC_INLINE SimpleTemplate<T>::badNestedTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: 'badNestedTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename T> template <typename... Ts> LIBC_INLINE void SimpleTemplate<T>::goodNestedVariadicTemplate() {}

template <typename T> template <typename... Ts> inline void SimpleTemplate<T>::badNestedVariadicTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: 'badNestedVariadicTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename T> template <typename... Ts> LIBC_INLINE inline void SimpleTemplate<T>::badNestedVariadicTemplate() {}

template <typename T> template <typename... Ts> void LIBC_INLINE SimpleTemplate<T>::badNestedVariadicTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: 'badNestedVariadicTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename... Ts>
struct VariadicTemplate {
  void goodVariadicTemplate();
  void badVariadicTemplate();
  void badVariadicTemplateWrongLocation();

  template <typename U>
  void goodNestedTemplate();
  template <typename U>
  void badNestedTemplate();
  template <typename U>
  void badNestedTemplateWrongLocation();

  template <typename... Us>
  void goodNestedVariadicTemplate();
  template <typename... Us>
  void badNestedVariadicTemplate();
  template <typename... Us>
  void badNestedVariadicTemplateWrongLocation();
};

template <typename... Ts> LIBC_INLINE void VariadicTemplate<Ts...>::goodVariadicTemplate() {}

template <typename... Ts> inline void VariadicTemplate<Ts...>::badVariadicTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'badVariadicTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename... Ts> LIBC_INLINE inline void VariadicTemplate<Ts...>::badVariadicTemplate() {}

template <typename... Ts> void LIBC_INLINE VariadicTemplate<Ts...>::badVariadicTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'badVariadicTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename... Ts> template <typename U> LIBC_INLINE void VariadicTemplate<Ts...>::goodNestedTemplate() {}

template <typename... Ts> template <typename U> inline void VariadicTemplate<Ts...>::badNestedTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: 'badNestedTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename... Ts> template <typename U> LIBC_INLINE inline void VariadicTemplate<Ts...>::badNestedTemplate() {}

template <typename... Ts> template <typename U> void LIBC_INLINE VariadicTemplate<Ts...>::badNestedTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: 'badNestedTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename... Ts> template <typename... Us> LIBC_INLINE void VariadicTemplate<Ts...>::goodNestedVariadicTemplate() {}

template <typename... Ts> template <typename... Us> inline void VariadicTemplate<Ts...>::badNestedVariadicTemplate() {}
// CHECK-MESSAGES: :[[@LINE-1]]:53: warning: 'badNestedVariadicTemplate' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
// CHECK-FIXES: template <typename... Ts> template <typename... Us> LIBC_INLINE inline void VariadicTemplate<Ts...>::badNestedVariadicTemplate() {}

template <typename... Ts> template <typename... Us> void LIBC_INLINE VariadicTemplate<Ts...>::badNestedVariadicTemplateWrongLocation() {}
// CHECK-MESSAGES: :[[@LINE-1]]:53: warning: 'badNestedVariadicTemplateWrongLocation' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

template <typename T>
void goodWeirdFormatting();
template <typename T>
void badWeirdFormatting();

template <typename T>LIBC_INLINE void goodWeirdFormatting() {}

template <typename T>void LIBC_INLINE badWeirdFormatting() {}
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 'badWeirdFormatting' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

} // namespace issue_62746

} // namespace __llvm_libc

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_LLVMLIBC_INLINEFUNCTIONDECL_H
