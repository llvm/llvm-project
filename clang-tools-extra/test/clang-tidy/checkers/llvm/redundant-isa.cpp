// RUN: %check_clang_tidy -std=c++17-or-later %s llvm-redundant-casting %t -- -- -fno-delayed-template-parsing

namespace llvm {
#define ISA_FUNCTION(name)                                           \
template <typename ...To, typename From>                             \
[[nodiscard]] inline bool name(const From &Val) {                    \
  return true;                                                       \
}                                                                    \
template <typename ...To, typename From>                             \
[[nodiscard]] inline bool name(const From *Val) {                    \
  return true;                                                       \
}

ISA_FUNCTION(isa)
ISA_FUNCTION(isa_and_present)
ISA_FUNCTION(isa_and_nonnull)
} // namespace llvm

struct A {};
struct B : A {};
A &getA();

void testIsa(A& value) {
  bool b1 = llvm::isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'A'
  (void)b1;
}

void testIsaAndPresent(A& value) {
  bool b2 = llvm::isa_and_present<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa_and_present' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:38: note: source expression has type 'A'
  (void)b2;
}

void testIsaAndNonnull(A& value) {
  bool b3 = llvm::isa_and_nonnull<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa_and_nonnull' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:38: note: source expression has type 'A'
  (void)b3;
}

void testIsaNonDeclRef() {
  bool b4 = llvm::isa<A>((getA()));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'A'
  (void)b4;
}

void testUpcast(B& value) {
  bool b5 = llvm::isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has type 'B', which is a subtype of 'A'
  (void)b5;
}

void testDowncast(A& value) {
  bool b6 = llvm::isa<B>(value);
  (void)b6;
}

namespace llvm {
void testIsaInLLVM(A& value) {
  bool b7 = isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:20: note: source expression has type 'A'
  (void)b7;
}
} // namespace llvm

void testIsaPointer(A* value) {
  bool b8 = llvm::isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: source expression has pointee type 'A'
  (void)b8;
}

template <typename T>
void testIsaTemplateIgnore(T* value) {
  bool b9 = llvm::isa<A>(value);
  (void)b9;
}
template void testIsaTemplateIgnore<A>(A *value);

template <typename T>
struct testIsaTemplateIgnoreStruct {
  void method(T* value) {
    bool b10 = llvm::isa<A>(value);
    (void)b10;
  }
};

void call(testIsaTemplateIgnoreStruct<A> s, A *a) {
  s.method(a);
}

template <typename T>
void testIsaTemplateTrigger1(T* value) {
  bool b11 = llvm::isa<T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:27: note: source expression has pointee type 'T'
  (void)b11;
}

template <typename T>
void testIsaTemplateTrigger2(A* value, T other) {
  bool b12 = llvm::isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:27: note: source expression has pointee type 'A'
  (void)b12; (void)other;
}

void testIsaConst(const A& value) {
  bool b13 = llvm::isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:27: note: source expression has type 'A'
  (void)b13;
}

void testIsaConstPointer(const A* value) {
  bool b14 = llvm::isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:27: note: source expression has pointee type 'A'
  (void)b14;
}

void testIsaMulti(A& value) {
  bool b15 = llvm::isa<int, A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:32: note: source expression has type 'A'
  (void)b15;
}

void testIsaMultiUpcast(B& value) {
  bool b16 = llvm::isa<int, A, B>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:35: note: source expression has type 'B', which is a subtype of 'A'
  (void)b16;
}

template <typename T>
void testIsaTemplateMulti(const T* value) {
  bool b17 = llvm::isa<A, T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:30: note: source expression has pointee type 'T'
  (void)b17;
}

namespace llvm {
namespace magic {
template <typename T>
void testIsaImplicitlyLLVMUnresolve(T* value) {
  bool b18 = isa<T>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:21: note: source expression has pointee type 'T'
  (void)b18;
}
} // namespace magic
} // namespace llvm

namespace mlir {
using llvm::isa;
void testIsaUsing(A& value) {
  bool b20 = isa<A>(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: call to 'isa' always succeeds [llvm-redundant-casting]
  // CHECK-MESSAGES: :[[@LINE-2]]:21: note: source expression has type 'A'
  (void)b20;
}
}

ISA_FUNCTION(isa)

void testIsaNonLLVM(A& value) {
  bool b19 = isa<A>(value);
  (void)b19;
}

