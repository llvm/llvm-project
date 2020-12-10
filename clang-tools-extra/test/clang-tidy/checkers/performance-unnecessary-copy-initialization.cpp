// RUN: %check_clang_tidy %s performance-unnecessary-copy-initialization %t

struct ExpensiveToCopyType {
  ExpensiveToCopyType();
  virtual ~ExpensiveToCopyType();
  const ExpensiveToCopyType &reference() const;
  void nonConstMethod();
  bool constMethod() const;
};

struct TrivialToCopyType {
  const TrivialToCopyType &reference() const;
};

struct WeirdCopyCtorType {
  WeirdCopyCtorType();
  WeirdCopyCtorType(const WeirdCopyCtorType &w, bool oh_yes = true);

  void nonConstMethod();
  bool constMethod() const;
};

ExpensiveToCopyType global_expensive_to_copy_type;

const ExpensiveToCopyType &ExpensiveTypeReference();
const ExpensiveToCopyType &freeFunctionWithArg(const ExpensiveToCopyType &);
const ExpensiveToCopyType &freeFunctionWithDefaultArg(
    const ExpensiveToCopyType *arg = nullptr);
const TrivialToCopyType &TrivialTypeReference();

void mutate(ExpensiveToCopyType &);
void mutate(ExpensiveToCopyType *);
void useAsConstPointer(const ExpensiveToCopyType *);
void useAsConstReference(const ExpensiveToCopyType &);
void useByValue(ExpensiveToCopyType);

void PositiveFunctionCall() {
  const auto AutoAssigned = ExpensiveTypeReference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned' is copy-constructed from a const reference; consider making it a const reference [performance-unnecessary-copy-initialization]
  // CHECK-FIXES: const auto& AutoAssigned = ExpensiveTypeReference();
  const auto AutoCopyConstructed(ExpensiveTypeReference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoCopyConstructed'
  // CHECK-FIXES: const auto& AutoCopyConstructed(ExpensiveTypeReference());
  const ExpensiveToCopyType VarAssigned = ExpensiveTypeReference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarAssigned'
  // CHECK-FIXES:   const ExpensiveToCopyType& VarAssigned = ExpensiveTypeReference();
  const ExpensiveToCopyType VarCopyConstructed(ExpensiveTypeReference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarCopyConstructed'
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(ExpensiveTypeReference());
}

void PositiveMethodCallConstReferenceParam(const ExpensiveToCopyType &Obj) {
  const auto AutoAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoCopyConstructed'
  // CHECK-FIXES: const auto& AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarAssigned'
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarCopyConstructed'
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(Obj.reference());
}

void PositiveMethodCallConstParam(const ExpensiveToCopyType Obj) {
  const auto AutoAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoCopyConstructed'
  // CHECK-FIXES: const auto& AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarAssigned'
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarCopyConstructed'
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(Obj.reference());
}

void PositiveMethodCallConstPointerParam(const ExpensiveToCopyType *const Obj) {
  const auto AutoAssigned = Obj->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = Obj->reference();
  const auto AutoCopyConstructed(Obj->reference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoCopyConstructed'
  // CHECK-FIXES: const auto& AutoCopyConstructed(Obj->reference());
  const ExpensiveToCopyType VarAssigned = Obj->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarAssigned'
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = Obj->reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj->reference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable 'VarCopyConstructed'
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(Obj->reference());
}

void PositiveLocalConstValue() {
  const ExpensiveToCopyType Obj;
  const auto UnnecessaryCopy = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'UnnecessaryCopy'
  // CHECK-FIXES: const auto& UnnecessaryCopy = Obj.reference();
}

void PositiveLocalConstRef() {
  const ExpensiveToCopyType Obj;
  const ExpensiveToCopyType &ConstReference = Obj.reference();
  const auto UnnecessaryCopy = ConstReference.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'UnnecessaryCopy'
  // CHECK-FIXES: const auto& UnnecessaryCopy = ConstReference.reference();
}

void PositiveLocalConstPointer() {
  const ExpensiveToCopyType Obj;
  const ExpensiveToCopyType *const ConstPointer = &Obj;
  const auto UnnecessaryCopy = ConstPointer->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'UnnecessaryCopy'
  // CHECK-FIXES: const auto& UnnecessaryCopy = ConstPointer->reference();
}

void NegativeFunctionCallTrivialType() {
  const auto AutoAssigned = TrivialTypeReference();
  const auto AutoCopyConstructed(TrivialTypeReference());
  const TrivialToCopyType VarAssigned = TrivialTypeReference();
  const TrivialToCopyType VarCopyConstructed(TrivialTypeReference());
}

void NegativeStaticLocalVar(const ExpensiveToCopyType &Obj) {
  static const auto StaticVar = Obj.reference();
}

void PositiveFunctionCallExpensiveTypeNonConstVariable() {
  auto AutoAssigned = ExpensiveTypeReference();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: the variable 'AutoAssigned' is copy-constructed from a const reference but is only used as const reference; consider making it a const reference [performance-unnecessary-copy-initialization]
  // CHECK-FIXES: const auto& AutoAssigned = ExpensiveTypeReference();
  auto AutoCopyConstructed(ExpensiveTypeReference());
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: the variable 'AutoCopyConstructed'
  // CHECK-FIXES: const auto& AutoCopyConstructed(ExpensiveTypeReference());
  ExpensiveToCopyType VarAssigned = ExpensiveTypeReference();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: the variable 'VarAssigned'
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = ExpensiveTypeReference();
  ExpensiveToCopyType VarCopyConstructed(ExpensiveTypeReference());
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: the variable 'VarCopyConstructed'
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(ExpensiveTypeReference());
}

void positiveNonConstVarInCodeBlock(const ExpensiveToCopyType &Obj) {
  {
    auto Assigned = Obj.reference();
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: the variable 'Assigned'
    // CHECK-FIXES: const auto& Assigned = Obj.reference();
    Assigned.reference();
    useAsConstReference(Assigned);
    useByValue(Assigned);
  }
}

void negativeNonConstVarWithNonConstUse(const ExpensiveToCopyType &Obj) {
  {
    auto NonConstInvoked = Obj.reference();
    // CHECK-FIXES: auto NonConstInvoked = Obj.reference();
    NonConstInvoked.nonConstMethod();
  }
  {
    auto Reassigned = Obj.reference();
    // CHECK-FIXES: auto Reassigned = Obj.reference();
    Reassigned = ExpensiveToCopyType();
  }
  {
    auto MutatedByReference = Obj.reference();
    // CHECK-FIXES: auto MutatedByReference = Obj.reference();
    mutate(MutatedByReference);
  }
  {
    auto MutatedByPointer = Obj.reference();
    // CHECK-FIXES: auto MutatedByPointer = Obj.reference();
    mutate(&MutatedByPointer);
  }
}

void PositiveMethodCallNonConstRefNotModified(ExpensiveToCopyType &Obj) {
  const auto AutoAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = Obj.reference();
}

void NegativeMethodCallNonConstRefIsModified(ExpensiveToCopyType &Obj) {
  const auto AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
  mutate(&Obj);
}

void PositiveMethodCallNonConstNotModified(ExpensiveToCopyType Obj) {
  const auto AutoAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = Obj.reference();
}

void NegativeMethodCallNonConstValueArgumentIsModified(ExpensiveToCopyType Obj) {
  Obj.nonConstMethod();
  const auto AutoAssigned = Obj.reference();
}

void PositiveMethodCallNonConstPointerNotModified(ExpensiveToCopyType *const Obj) {
  const auto AutoAssigned = Obj->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = Obj->reference();
  Obj->constMethod();
}

void NegativeMethodCallNonConstPointerIsModified(ExpensiveToCopyType *const Obj) {
  const auto AutoAssigned = Obj->reference();
  const auto AutoCopyConstructed(Obj->reference());
  const ExpensiveToCopyType VarAssigned = Obj->reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj->reference());
  mutate(Obj);
}

void PositiveLocalVarIsNotModified() {
  ExpensiveToCopyType LocalVar;
  const auto AutoAssigned = LocalVar.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned'
  // CHECK-FIXES: const auto& AutoAssigned = LocalVar.reference();
}

void NegativeLocalVarIsModified() {
  ExpensiveToCopyType Obj;
  const auto AutoAssigned = Obj.reference();
  Obj = AutoAssigned;
}

struct NegativeConstructor {
  NegativeConstructor(const ExpensiveToCopyType &Obj) : Obj(Obj) {}
  ExpensiveToCopyType Obj;
};

#define UNNECESSARY_COPY_INIT_IN_MACRO_BODY(TYPE)                              \
  void functionWith##TYPE(const TYPE &T) {                                     \
    auto AssignedInMacro = T.reference();                                      \
  }                                                                            \
// Ensure fix is not applied.
// CHECK-FIXES: auto AssignedInMacro = T.reference();

UNNECESSARY_COPY_INIT_IN_MACRO_BODY(ExpensiveToCopyType)
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: the variable 'AssignedInMacro' is copy-constructed

#define UNNECESSARY_COPY_INIT_IN_MACRO_ARGUMENT(ARGUMENT) ARGUMENT

void PositiveMacroArgument(const ExpensiveToCopyType &Obj) {
  UNNECESSARY_COPY_INIT_IN_MACRO_ARGUMENT(auto CopyInMacroArg = Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:48: warning: the variable 'CopyInMacroArg' is copy-constructed
  // Ensure fix is not applied.
  // CHECK-FIXES: auto CopyInMacroArg = Obj.reference()
}

void PositiveLocalCopyConstMethodInvoked() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType copy_1 = orig;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy_1' of the variable 'orig' is never modified; consider avoiding the copy [performance-unnecessary-copy-initialization]
  // CHECK-FIXES: const ExpensiveToCopyType& copy_1 = orig;
  copy_1.constMethod();
  orig.constMethod();
}

void PositiveLocalCopyUsingExplicitCopyCtor() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType copy_2(orig);
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy_2'
  // CHECK-FIXES: const ExpensiveToCopyType& copy_2(orig);
  copy_2.constMethod();
  orig.constMethod();
}

void PositiveLocalCopyCopyIsArgument(const ExpensiveToCopyType &orig) {
  ExpensiveToCopyType copy_3 = orig;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy_3'
  // CHECK-FIXES: const ExpensiveToCopyType& copy_3 = orig;
  copy_3.constMethod();
}

void PositiveLocalCopyUsedAsConstRef() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType copy_4 = orig;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy_4'
  // CHECK-FIXES: const ExpensiveToCopyType& copy_4 = orig;
  useAsConstReference(orig);
}

void PositiveLocalCopyTwice() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType copy_5 = orig;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy_5'
  // CHECK-FIXES: const ExpensiveToCopyType& copy_5 = orig;
  ExpensiveToCopyType copy_6 = copy_5;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy_6'
  // CHECK-FIXES: const ExpensiveToCopyType& copy_6 = copy_5;
  copy_5.constMethod();
  copy_6.constMethod();
  orig.constMethod();
}


void PositiveLocalCopyWeirdCopy() {
  WeirdCopyCtorType orig;
  WeirdCopyCtorType weird_1(orig);
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: local copy 'weird_1'
  // CHECK-FIXES: const WeirdCopyCtorType& weird_1(orig);
  weird_1.constMethod();

  WeirdCopyCtorType weird_2 = orig;
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: local copy 'weird_2'
  // CHECK-FIXES: const WeirdCopyCtorType& weird_2 = orig;
  weird_2.constMethod();
}

void NegativeLocalCopySimpleTypes() {
  int i1 = 0;
  int i2 = i1;
}

void NegativeLocalCopyCopyIsModified() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType neg_copy_1 = orig;
  neg_copy_1.nonConstMethod();
}

void NegativeLocalCopyOriginalIsModified() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType neg_copy_2 = orig;
  orig.nonConstMethod();
}

void NegativeLocalCopyUsedAsRefArg() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType neg_copy_3 = orig;
  mutate(neg_copy_3);
}

void NegativeLocalCopyUsedAsPointerArg() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType neg_copy_4 = orig;
  mutate(&neg_copy_4);
}

void NegativeLocalCopyCopyFromGlobal() {
  ExpensiveToCopyType neg_copy_5 = global_expensive_to_copy_type;
}

void NegativeLocalCopyCopyToStatic() {
  ExpensiveToCopyType orig;
  static ExpensiveToCopyType neg_copy_6 = orig;
}

void NegativeLocalCopyNonConstInForLoop() {
  ExpensiveToCopyType orig;
  for (ExpensiveToCopyType neg_copy_7 = orig; orig.constMethod();
       orig.nonConstMethod()) {
    orig.constMethod();
  }
}

void NegativeLocalCopyWeirdNonCopy() {
  WeirdCopyCtorType orig;
  WeirdCopyCtorType neg_weird_1(orig, false);
  WeirdCopyCtorType neg_weird_2(orig, true);
}
void WarningOnlyMultiDeclStmt() {
  ExpensiveToCopyType orig;
  ExpensiveToCopyType copy = orig, copy2;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: local copy 'copy' of the variable 'orig' is never modified; consider avoiding the copy [performance-unnecessary-copy-initialization]
  // CHECK-FIXES: ExpensiveToCopyType copy = orig, copy2;
}

class Element {};
class Container {
public:
  class Iterator {
  public:
    void operator++();
    Element operator*();
    bool operator!=(const Iterator &);
    WeirdCopyCtorType c;
  };
  const Iterator &begin() const;
  const Iterator &end() const;
};

void implicitVarFalsePositive() {
  for (const Element &E : Container()) {
  }
}

// This should not trigger the check as the argument could introduce an alias.
void negativeInitializedFromFreeFunctionWithArg() {
  ExpensiveToCopyType Orig;
  const ExpensiveToCopyType Copy = freeFunctionWithArg(Orig);
}

void negativeInitializedFromFreeFunctionWithDefaultArg() {
  const ExpensiveToCopyType Copy = freeFunctionWithDefaultArg();
}

void negativeInitialzedFromFreeFunctionWithNonDefaultArg() {
  ExpensiveToCopyType Orig;
  const ExpensiveToCopyType Copy = freeFunctionWithDefaultArg(&Orig);
}

namespace std {
inline namespace __1 {

template <class>
class function;
template <class R, class... ArgTypes>
class function<R(ArgTypes...)> {
public:
  function();
  function(const function &Other);
  R operator()(ArgTypes... Args) const;
};

} // namespace __1
} // namespace std

void negativeStdFunction() {
  std::function<int()> Orig;
  std::function<int()> Copy = Orig;
  int i = Orig();
}

using Functor = std::function<int()>;

void negativeAliasedStdFunction() {
  Functor Orig;
  Functor Copy = Orig;
  int i = Orig();
}

typedef std::function<int()> TypedefFunc;

void negativeTypedefedStdFunction() {
  TypedefFunc Orig;
  TypedefFunc Copy = Orig;
  int i = Orig();
}

namespace fake {
namespace std {
template <class R, class... Args>
struct function {
  // Custom copy constructor makes it expensive to copy;
  function(const function &);
};
} // namespace std

void positiveFakeStdFunction(std::function<void(int)> F) {
  auto Copy = F;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: local copy 'Copy' of the variable 'F' is never modified;
  // CHECK-FIXES: const auto& Copy = F;
}

} // namespace fake

void positiveInvokedOnStdFunction(
    std::function<void(const ExpensiveToCopyType &)> Update,
    const ExpensiveToCopyType Orig) {
  auto Copy = Orig.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: the variable 'Copy' is copy-constructed from a const reference
  // CHECK-FIXES: const auto& Copy = Orig.reference();
  Update(Copy);
}

void negativeInvokedOnStdFunction(
    std::function<void(ExpensiveToCopyType &)> Update,
    const ExpensiveToCopyType Orig) {
  auto Copy = Orig.reference();
  Update(Copy);
}
