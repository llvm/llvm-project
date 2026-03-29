// RUN: %check_clang_tidy -std=c++17-or-later %s llvm-redundant-casting %t -- -- -fno-delayed-template-parsing

namespace llvm {
#define CAST_FUNCTION(name)                                          \
template <typename To, typename From>                                \
[[nodiscard]] inline decltype(auto) name(const From &Val) {          \
  return static_cast<const To&>(Val);                                \
}                                                                    \
template <typename To, typename From>                                \
[[nodiscard]] inline decltype(auto) name(From &Val) {                \
  return static_cast<To&>(Val);                                      \
}                                                                    \
template <typename To, typename From>                                \
[[nodiscard]] inline decltype(auto) name(const From *Val) {          \
  return static_cast<const To*>(Val);                                \
}                                                                    \
template <typename To, typename From>                                \
[[nodiscard]] inline decltype(auto) name(From *Val) {                \
  return static_cast<To*>(Val);                                      \
}
CAST_FUNCTION(cast)
CAST_FUNCTION(dyn_cast)
CAST_FUNCTION(cast_or_null)
CAST_FUNCTION(cast_if_present)
CAST_FUNCTION(dyn_cast_or_null)
CAST_FUNCTION(dyn_cast_if_present)
}

struct A {};
struct B : A {};
A &getA();

void testCast(A& value) {
  A& a1 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a1 = value;
  (void)a1;
}

void testDynCast(A& value) {
  A& a2 = llvm::dyn_cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'dyn_cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:29: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a2 = value;
  (void)a2;
}

void testCastOrNull(A& value) {
  A& a3 = llvm::cast_or_null<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast_or_null' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:33: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a3 = value;
  (void)a3;
}

void testCastIfPresent(A& value) {
  A& a4 = llvm::cast_if_present<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast_if_present' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:36: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a4 = value;
  (void)a4;
}

void testDynCastOrNull(A& value) {
  A& a5 = llvm::dyn_cast_or_null<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'dyn_cast_or_null' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:37: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a5 = value;
  (void)a5;
}

void testDynCastIfPresent(A& value) {
  A& a6 = llvm::dyn_cast_if_present<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'dyn_cast_if_present' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:40: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a6 = value;
  (void)a6;
}

void testCastNonDeclRef() {
  A& a7 = llvm::cast<A>((getA()));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: source expression '(getA())' has type 'A'
  // CHECK-FIXES: A& a7 = (getA());
  (void)a7;
}

void testUpcast(B& value) {
  A& a8 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: source expression 'value' has type 'B'
  // CHECK-MESSAGES: :[[@LINE-3]]:25: note: 'B' is a subtype of 'A'
  // CHECK-FIXES: A& a8 = value;
  (void)a8;
}

void testDowncast(A& value) {
  B& a9 = llvm::cast<B>(value);
  // CHECK-MESSAGES-NOT: warning
  // CHECK-FIXES-NOT: A& a9 = value;
  (void)a9;
}

namespace llvm {
void testCastInLLVM(A& value) {
  A& a11 = cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:20: note: source expression 'value' has type 'A'
  // CHECK-FIXES: A& a11 = value;
  (void)a11;
}
} // namespace llvm

void testCastPointer(A* value) {
  A *a12 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression 'value' has type 'A *'
  // CHECK-FIXES: A *a12 = value;
  (void)a12;
}

template <typename T>
void testCastTemplateIgnore(T* value) {
  A *a13 = llvm::cast<A>(value);
  // CHECK-MESSAGES-NOT: warning
  // CHECK-FIXES-NOT: A *a13 = value;
  (void)a13;
}
template void testCastTemplateIgnore<A>(A *value);

template <typename T>
struct testCastTemplateIgnoreStruct {
  void method(T* value) {
    A *a14 = llvm::cast<A>(value);
    // CHECK-MESSAGES-NOT: warning
    // CHECK-FIXES-NOT: A *a14 = value;
    (void)a14;
  }
};

void call(testCastTemplateIgnoreStruct<A> s, A *a) {
  s.method(a);
}

template <typename T>
void testCastTemplateTrigger1(T* value) {
  T *a15 = llvm::cast<T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression 'value' has type 'T *'
  // CHECK-FIXES: T *a15 = value;
  (void)a15;
}

template <typename T>
void testCastTemplateTrigger2(A* value, T other) {
  A *a16 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression 'value' has type 'A *'
  // CHECK-FIXES: A *a16 = value;
  (void)a16; (void) other;
}

