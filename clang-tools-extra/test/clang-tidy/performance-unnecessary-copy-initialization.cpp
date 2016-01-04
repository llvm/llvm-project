// RUN: %check_clang_tidy %s performance-unnecessary-copy-initialization %t

struct ExpensiveToCopyType {
  ExpensiveToCopyType() {}
  virtual ~ExpensiveToCopyType() {}
  const ExpensiveToCopyType &reference() const { return *this; }
};

struct TrivialToCopyType {
  const TrivialToCopyType &reference() const { return *this; }
};

const ExpensiveToCopyType &ExpensiveTypeReference() {
  static const ExpensiveToCopyType *Type = new ExpensiveToCopyType();
  return *Type;
}

const TrivialToCopyType &TrivialTypeReference() {
  static const TrivialToCopyType *Type = new TrivialToCopyType();
  return *Type;
}

void PositiveFunctionCall() {
  const auto AutoAssigned = ExpensiveTypeReference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'AutoAssigned' is copy-constructed from a const reference; consider making it a const reference [performance-unnecessary-copy-initialization]
  // CHECK-FIXES: const auto& AutoAssigned = ExpensiveTypeReference();
  const auto AutoCopyConstructed(ExpensiveTypeReference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoCopyConstructed(ExpensiveTypeReference());
  const ExpensiveToCopyType VarAssigned = ExpensiveTypeReference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES:   const ExpensiveToCopyType& VarAssigned = ExpensiveTypeReference();
  const ExpensiveToCopyType VarCopyConstructed(ExpensiveTypeReference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(ExpensiveTypeReference());
}

void PositiveMethodCallConstReferenceParam(const ExpensiveToCopyType &Obj) {
  const auto AutoAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(Obj.reference());
}

void PositiveMethodCallConstParam(const ExpensiveToCopyType Obj) {
  const auto AutoAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(Obj.reference());
}

void PositiveMethodCallConstPointerParam(const ExpensiveToCopyType *const Obj) {
  const auto AutoAssigned = Obj->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoAssigned = Obj->reference();
  const auto AutoCopyConstructed(Obj->reference());
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& AutoCopyConstructed(Obj->reference());
  const ExpensiveToCopyType VarAssigned = Obj->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarAssigned = Obj->reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj->reference());
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: the const qualified variable
  // CHECK-FIXES: const ExpensiveToCopyType& VarCopyConstructed(Obj->reference());
}

void PositiveLocalConstValue() {
  const ExpensiveToCopyType Obj;
  const auto UnnecessaryCopy = Obj.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& UnnecessaryCopy = Obj.reference();
}

void PositiveLocalConstRef() {
  const ExpensiveToCopyType Obj;
  const ExpensiveToCopyType &ConstReference = Obj.reference();
  const auto UnnecessaryCopy = ConstReference.reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
  // CHECK-FIXES: const auto& UnnecessaryCopy = ConstReference.reference();
}

void PositiveLocalConstPointer() {
  const ExpensiveToCopyType Obj;
  const ExpensiveToCopyType *const ConstPointer = &Obj;
  const auto UnnecessaryCopy = ConstPointer->reference();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable
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

void NegativeFunctionCallExpensiveTypeNonConstVariable() {
  auto AutoAssigned = ExpensiveTypeReference();
  auto AutoCopyConstructed(ExpensiveTypeReference());
  ExpensiveToCopyType VarAssigned = ExpensiveTypeReference();
  ExpensiveToCopyType VarCopyConstructed(ExpensiveTypeReference());
}

void NegativeMethodCallNonConstRef(ExpensiveToCopyType &Obj) {
  const auto AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
}

void NegativeMethodCallNonConst(ExpensiveToCopyType Obj) {
  const auto AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
}

void NegativeMethodCallNonConstPointer(ExpensiveToCopyType *const Obj) {
  const auto AutoAssigned = Obj->reference();
  const auto AutoCopyConstructed(Obj->reference());
  const ExpensiveToCopyType VarAssigned = Obj->reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj->reference());
}

void NegativeObjIsNotParam() {
  ExpensiveToCopyType Obj;
  const auto AutoAssigned = Obj.reference();
  const auto AutoCopyConstructed(Obj.reference());
  const ExpensiveToCopyType VarAssigned = Obj.reference();
  const ExpensiveToCopyType VarCopyConstructed(Obj.reference());
}

struct NegativeConstructor {
  NegativeConstructor(const ExpensiveToCopyType &Obj) : Obj(Obj) {}
  ExpensiveToCopyType Obj;
};
