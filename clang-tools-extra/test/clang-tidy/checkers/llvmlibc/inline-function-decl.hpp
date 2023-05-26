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
  return a + b;
}

inline unsigned long addul(unsigned long a, unsigned long b) {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'addul' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
  return a + b;
}

class  MyClass {
  int A;
public:
  MyClass() : A(123) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'MyClass' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

  LIBC_INLINE MyClass(int V) : A(V) {}

  constexpr operator int() const { return A; }
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator int' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]

  LIBC_INLINE bool operator==(const MyClass &RHS) {
    return RHS.A == A;
  }

  static int getVal(const MyClass &V) {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'getVal' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
    return V.A;
  }

  LIBC_INLINE static void setVal(MyClass &V, int A) {
    V.A = A;
  }

  constexpr static int addInt(MyClass &V, int A) {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'addInt' must be tagged with the LIBC_INLINE macro; the macro should be placed at the beginning of the declaration [llvmlibc-inline-function-decl]
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

} // namespace __llvm_libc

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_LLVMLIBC_INLINEFUNCTIONDECL_H
