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
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: source expression has type 'A'
  // CHECK-FIXES: A& a1 = value;
  (void)a1;
}

void testDynCast(A& value) {
  A& a2 = llvm::dyn_cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'dyn_cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:29: note: source expression has type 'A'
  // CHECK-FIXES: A& a2 = value;
  (void)a2;
}

void testCastOrNull(A& value) {
  A& a3 = llvm::cast_or_null<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast_or_null' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:33: note: source expression has type 'A'
  // CHECK-FIXES: A& a3 = value;
  (void)a3;
}

void testCastIfPresent(A& value) {
  A& a4 = llvm::cast_if_present<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast_if_present' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:36: note: source expression has type 'A'
  // CHECK-FIXES: A& a4 = value;
  (void)a4;
}

void testDynCastOrNull(A& value) {
  A& a5 = llvm::dyn_cast_or_null<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'dyn_cast_or_null' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:37: note: source expression has type 'A'
  // CHECK-FIXES: A& a5 = value;
  (void)a5;
}

void testDynCastIfPresent(A& value) {
  A& a6 = llvm::dyn_cast_if_present<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'dyn_cast_if_present' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:40: note: source expression has type 'A'
  // CHECK-FIXES: A& a6 = value;
  (void)a6;
}

void testCastNonDeclRef() {
  A& a7 = llvm::cast<A>((getA()));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: source expression has type 'A'
  // CHECK-FIXES: A& a7 = (getA());
  (void)a7;
}

void testUpcast(B& value) {
  A& a8 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: source expression has type 'B', which is a subtype of 'A'
  // CHECK-FIXES: A& a8 = value;
  (void)a8;
}

void testDowncast(A& value) {
  B& a9 = llvm::cast<B>(value);
  // CHECK-MESSAGES-NOT: warning
  // CHECK-FIXES-NOT: A& a9 = value;
  (void)a9;
}

struct C : B {};

void testUpcastTransitive(C& value) {
  A& a10 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'C', which is a subtype of 'A'
  // CHECK-FIXES: A& a10 = value;
  (void)a10;
}

namespace llvm {
void testCastInLLVM(A& value) {
  A& a11 = cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:20: note: source expression has type 'A'
  // CHECK-FIXES: A& a11 = value;
  (void)a11;
}
} // namespace llvm

void testCastPointer(A* value) {
  A *a12 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has pointee type 'A'
  // CHECK-FIXES: A *a12 = value;
  (void)a12;
}

template <typename T>
void testCastTemplateIgnore(T* value) {
  A *a13 = llvm::cast<A>(value);
  (void)a13;
}
template void testCastTemplateIgnore<A>(A *value);

template <typename T>
struct testCastTemplateIgnoreStruct {
  void method(T* value) {
    A *a14 = llvm::cast<A>(value);
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
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has pointee type 'T'
  // CHECK-FIXES: T *a15 = value;
  (void)a15;
}

template <typename T>
void testCastTemplateTrigger2(A* value, T other) {
  A *a16 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has pointee type 'A'
  // CHECK-FIXES: A *a16 = value;
  (void)a16; (void) other;
}

void testCastConst(const A& value) {
  const A& a17 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:32: note: source expression has type 'A'
  // CHECK-FIXES: const A& a17 = value;
  (void)a17;
}

void testCastConstPointer(const A* value) {
  const A* a18 = llvm::cast<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:32: note: source expression has pointee type 'A'
  // CHECK-FIXES: const A* a18 = value;
  (void)a18;
}

void testCastExplicitDowncastImplicitUpcast(const A* value) {
  const A* a19 = llvm::cast<C>(value);
  (void)a19;
}

struct D : A {};
struct E {};
struct F : D, E {};

void testCastUpcastMultipleInheritance(F& value) {
  E& a20 = llvm::cast<E>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'F', which is a subtype of 'E'
  // CHECK-FIXES: E& a20 = value;
  (void)a20;
}

struct G : D, C {};

void testCastUpcastDiamondExplicit(G& value) {
  A& a21 = llvm::cast<A>(llvm::cast<C>(value));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'C', which is a subtype of 'A'
  // CHECK-FIXES: A& a21 = llvm::cast<C>(value);
  (void)a21;
}

void testCastUpcastDiamondImplicit(G& value) {
  A& a22 = llvm::cast<C>(value);
  (void)a22;
}

void testCastUpcastDiamondSingleSide(G& value) {
  C& a23 = llvm::cast<C>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'G', which is a subtype of 'C'
  // CHECK-FIXES: C& a23 = value;
  (void)a23;
}

struct H : virtual A {};
struct I : virtual A {};
struct J : H, I {};

void testCastUpcastDiamondVirtual(J& value) {
  A& a24 = llvm::cast<I>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'J', which is a subtype of 'I'
  // CHECK-FIXES: A& a24 = value;
  (void)a24;
}

template<typename T>
struct K {};

struct L : K<L> {};

void testCastCRTPUpcast(L& value) {
  K<L>& a24 = llvm::cast<K<L>>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:32: note: source expression has type 'L', which is a subtype of 'K<L>'
  // CHECK-FIXES: K<L>& a24 = value;
  (void)a24;
}

CAST_FUNCTION(cast)
CAST_FUNCTION(dyn_cast)

void testCastNonLLVM(A& value) {
  A& a25 = cast<A>(value);
  (void)a25;
}

void testDynCastNonLLVM(A& value) {
  A& a26 = dyn_cast<A>(value);
  (void)a26;
}

template<typename T>
void testCastNonLLVMUnresolved(T& value) {
  T& a27 = cast<T>(value);
  (void)a27;
}

namespace llvm {
namespace magic {
template<typename T>
void testCastImplicitlyLLVMUnresolved(T& value) {
  T& a28 = cast<T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant use of 'cast' [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:20: note: source expression has type 'T'
  // CHECK-FIXES: T& a28 = value;
  (void)a28;
}
} // namespace magic
} // namespace llvm

// FIXME: this cast is redundant since it's immediately undone by the implicit cast
void testCastUpdown(A& value) {
  A& a29 = cast<C>(value);
  (void)a29;
}
