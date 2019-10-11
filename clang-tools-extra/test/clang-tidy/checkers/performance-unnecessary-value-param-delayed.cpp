// RUN: %check_clang_tidy %s performance-unnecessary-value-param %t -- -- -fdelayed-template-parsing

struct ExpensiveToCopyType {
  const ExpensiveToCopyType & constReference() const {
    return *this;
  }
  void nonConstMethod();
  virtual ~ExpensiveToCopyType();
};

void mutate(ExpensiveToCopyType &);
void mutate(ExpensiveToCopyType *);
void useAsConstReference(const ExpensiveToCopyType &);
void useByValue(ExpensiveToCopyType);

// This class simulates std::pair<>. It is trivially copy constructible
// and trivially destructible, but not trivially copy assignable.
class SomewhatTrivial {
 public:
  SomewhatTrivial();
  SomewhatTrivial(const SomewhatTrivial&) = default;
  ~SomewhatTrivial() = default;
  SomewhatTrivial& operator=(const SomewhatTrivial&);
};

void positiveExpensiveConstValue(const ExpensiveToCopyType Obj);
// CHECK-FIXES: void positiveExpensiveConstValue(const ExpensiveToCopyType& Obj);
void positiveExpensiveConstValue(const ExpensiveToCopyType Obj) {
  // CHECK-MESSAGES: [[@LINE-1]]:60: warning: the const qualified parameter 'Obj' is copied for each invocation; consider making it a reference [performance-unnecessary-value-param]
  // CHECK-FIXES: void positiveExpensiveConstValue(const ExpensiveToCopyType& Obj) {
}

void positiveExpensiveValue(ExpensiveToCopyType Obj);
// CHECK-FIXES: void positiveExpensiveValue(const ExpensiveToCopyType& Obj);
void positiveExpensiveValue(ExpensiveToCopyType Obj) {
  // CHECK-MESSAGES: [[@LINE-1]]:49: warning: the parameter 'Obj' is copied for each invocation but only used as a const reference; consider making it a const reference [performance-unnecessary-value-param]
  // CHECK-FIXES: void positiveExpensiveValue(const ExpensiveToCopyType& Obj) {
  Obj.constReference();
  useAsConstReference(Obj);
  auto Copy = Obj;
  useByValue(Obj);
}

void positiveWithComment(const ExpensiveToCopyType /* important */ S);
// CHECK-FIXES: void positiveWithComment(const ExpensiveToCopyType& /* important */ S);
void positiveWithComment(const ExpensiveToCopyType /* important */ S) {
  // CHECK-MESSAGES: [[@LINE-1]]:68: warning: the const qualified
  // CHECK-FIXES: void positiveWithComment(const ExpensiveToCopyType& /* important */ S) {
}

void positiveUnnamedParam(const ExpensiveToCopyType) {
  // CHECK-MESSAGES: [[@LINE-1]]:52: warning: the const qualified parameter #1
  // CHECK-FIXES: void positiveUnnamedParam(const ExpensiveToCopyType&) {
}

void positiveAndNegative(const ExpensiveToCopyType ConstCopy, const ExpensiveToCopyType& ConstRef, ExpensiveToCopyType Copy);
// CHECK-FIXES: void positiveAndNegative(const ExpensiveToCopyType& ConstCopy, const ExpensiveToCopyType& ConstRef, const ExpensiveToCopyType& Copy);
void positiveAndNegative(const ExpensiveToCopyType ConstCopy, const ExpensiveToCopyType& ConstRef, ExpensiveToCopyType Copy) {
  // CHECK-MESSAGES: [[@LINE-1]]:52: warning: the const qualified parameter 'ConstCopy'
  // CHECK-MESSAGES: [[@LINE-2]]:120: warning: the parameter 'Copy'
  // CHECK-FIXES: void positiveAndNegative(const ExpensiveToCopyType& ConstCopy, const ExpensiveToCopyType& ConstRef, const ExpensiveToCopyType& Copy) {
}

struct PositiveConstValueConstructor {
  PositiveConstValueConstructor(const ExpensiveToCopyType ConstCopy) {}
  // CHECK-MESSAGES: [[@LINE-1]]:59: warning: the const qualified parameter 'ConstCopy'
  // CHECK-FIXES: PositiveConstValueConstructor(const ExpensiveToCopyType& ConstCopy) {}
};

template <typename T> void templateWithNonTemplatizedParameter(const ExpensiveToCopyType S, T V) {
  // CHECK-MESSAGES: [[@LINE-1]]:90: warning: the const qualified parameter 'S'
  // CHECK-FIXES: template <typename T> void templateWithNonTemplatizedParameter(const ExpensiveToCopyType& S, T V) {
}

void instantiated() {
  templateWithNonTemplatizedParameter(ExpensiveToCopyType(), ExpensiveToCopyType());
  templateWithNonTemplatizedParameter(ExpensiveToCopyType(), 5);
}

template <typename T> void negativeTemplateType(const T V) {
}

void negativeArray(const ExpensiveToCopyType[]) {
}

void negativePointer(ExpensiveToCopyType* Obj) {
}

void negativeConstPointer(const ExpensiveToCopyType* Obj) {
}

void negativeConstReference(const ExpensiveToCopyType& Obj) {
}

void negativeReference(ExpensiveToCopyType& Obj) {
}

void negativeUniversalReference(ExpensiveToCopyType&& Obj) {
}

void negativeSomewhatTrivialConstValue(const SomewhatTrivial Somewhat) {
}

void negativeSomewhatTrivialValue(SomewhatTrivial Somewhat) {
}

void negativeConstBuiltIn(const int I) {
}

void negativeValueBuiltIn(int I) {
}

void negativeValueIsMutatedByReference(ExpensiveToCopyType Obj) {
  mutate(Obj);
}

void negativeValueIsMutatatedByPointer(ExpensiveToCopyType Obj) {
  mutate(&Obj);
}

void negativeValueIsReassigned(ExpensiveToCopyType Obj) {
  Obj = ExpensiveToCopyType();
}

void negativeValueNonConstMethodIsCalled(ExpensiveToCopyType Obj) {
  Obj.nonConstMethod();
}

struct PositiveValueUnusedConstructor {
  PositiveValueUnusedConstructor(ExpensiveToCopyType Copy) {}
  // CHECK-MESSAGES: [[@LINE-1]]:54: warning: the parameter 'Copy'
  // CHECK-FIXES: PositiveValueUnusedConstructor(const ExpensiveToCopyType& Copy) {}
};

struct PositiveValueCopiedConstructor {
  PositiveValueCopiedConstructor(ExpensiveToCopyType Copy) : Field(Copy) {}
  // CHECK-MESSAGES: [[@LINE-1]]:54: warning: the parameter 'Copy'
  // CHECK-FIXES: PositiveValueCopiedConstructor(const ExpensiveToCopyType& Copy) : Field(Copy) {}
  ExpensiveToCopyType Field;
};

template <typename T>
struct Container {
  typedef const T & const_reference;
};

void NegativeTypedefParam(const Container<ExpensiveToCopyType>::const_reference Param) {
}

#define UNNECESSARY_VALUE_PARAM_IN_MACRO_BODY()         \
  void inMacro(const ExpensiveToCopyType T) {           \
  }                                                     \
// Ensure fix is not applied.
// CHECK-FIXES: void inMacro(const ExpensiveToCopyType T) {

UNNECESSARY_VALUE_PARAM_IN_MACRO_BODY()
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: the const qualified parameter 'T'

#define UNNECESSARY_VALUE_PARAM_IN_MACRO_ARGUMENT(ARGUMENT)     \
  ARGUMENT

UNNECESSARY_VALUE_PARAM_IN_MACRO_ARGUMENT(void inMacroArgument(const ExpensiveToCopyType InMacroArg) {})
// CHECK-MESSAGES: [[@LINE-1]]:90: warning: the const qualified parameter 'InMacroArg'
// CHECK-FIXES: void inMacroArgument(const ExpensiveToCopyType InMacroArg) {}

struct VirtualMethod {
  virtual ~VirtualMethod() {}
  virtual void handle(ExpensiveToCopyType T) const = 0;
};

struct NegativeOverriddenMethod : public VirtualMethod {
  void handle(ExpensiveToCopyType Overridden) const {
    // CHECK-FIXES: handle(ExpensiveToCopyType Overridden) const {
  }
};

struct NegativeDeletedMethod {
  ~NegativeDeletedMethod() {}
  NegativeDeletedMethod& operator=(NegativeDeletedMethod N) = delete;
  // CHECK-FIXES: NegativeDeletedMethod& operator=(NegativeDeletedMethod N) = delete;
};
