// RUN: %check_clang_tidy  -std=c++14-or-later %s performance-unnecessary-value-param %t

struct ExpensiveToCopyType {
  virtual ~ExpensiveToCopyType();
};

template <typename T> void templateWithNonTemplatizedParameter(const ExpensiveToCopyType S, T V) {
  // CHECK-MESSAGES: [[@LINE-1]]:90: warning: the const qualified parameter 'S'
  // CHECK-MESSAGES: [[@LINE-2]]:95: warning: the parameter 'V'
  // CHECK-FIXES: template <typename T> void templateWithNonTemplatizedParameter(const ExpensiveToCopyType& S, const T& V) {
}

void instantiatedWithExpensiveValue() {
  templateWithNonTemplatizedParameter(
      ExpensiveToCopyType(), ExpensiveToCopyType());
  templateWithNonTemplatizedParameter(ExpensiveToCopyType(), 5);
}

template <typename T> void templateWithNonTemplatizedParameterCheapTemplate(const ExpensiveToCopyType S, T V) {
  // CHECK-MESSAGES: [[@LINE-1]]:103: warning: the const qualified parameter 'S'
  // CHECK-FIXES: template <typename T> void templateWithNonTemplatizedParameterCheapTemplate(const ExpensiveToCopyType& S, T V) {
}

void instantiatedWithCheapValue() {
  templateWithNonTemplatizedParameterCheapTemplate(ExpensiveToCopyType(), 5);
}

template <typename T> void nonInstantiatedTemplateWithConstValue(const T S) {}
template <typename T> void nonInstantiatedTemplateWithNonConstValue(T S) {}

template <typename T> void instantiatedTemplateSpecialization(T NoSpecS) {}
template <> void instantiatedTemplateSpecialization<int>(int SpecSInt) {}
// Updating template specialization would also require to update the main
// template and other specializations. Such specializations may be
// spreaded across different translation units.
// For that reason we only issue a warning, but do not propose fixes.
template <>
void instantiatedTemplateSpecialization<ExpensiveToCopyType>(
    ExpensiveToCopyType SpecSExpensiveToCopy) {
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: the parameter 'SpecSExpensiveToCopy'
  // CHECK-FIXES-NOT: const T& NoSpecS
  // CHECK-FIXES-NOT: const int& SpecSInt
  // CHECK-FIXES-NOT: const ExpensiveToCopyType& SpecSExpensiveToCopy
}

void instantiatedTemplateSpecialization() {
  instantiatedTemplateSpecialization(ExpensiveToCopyType());
}

template <typename T> void instantiatedTemplateWithConstValue(const T S) {
  // CHECK-MESSAGES: [[@LINE-1]]:71: warning: the const qualified parameter 'S'
  // CHECK-FIXES: template <typename T> void instantiatedTemplateWithConstValue(const T& S) {
}

void instantiatedConstValue() {
  instantiatedTemplateWithConstValue(ExpensiveToCopyType());
}

template <typename T> void instantiatedTemplateWithNonConstValue(T S) {
  // CHECK-MESSAGES: [[@LINE-1]]:68: warning: the parameter 'S'
  // CHECK-FIXES: template <typename T> void instantiatedTemplateWithNonConstValue(const T& S) {
}

void instantiatedNonConstValue() {
  instantiatedTemplateWithNonConstValue(ExpensiveToCopyType());
}

void lambdaConstValue() {
  auto fn = [](const ExpensiveToCopyType S) {
    // CHECK-MESSAGES: [[@LINE-1]]:42: warning: the const qualified parameter 'S'
    // CHECK-FIXES: auto fn = [](const ExpensiveToCopyType& S) {
  };
  fn(ExpensiveToCopyType());
}

void lambdaNonConstValue() {
  auto fn = [](ExpensiveToCopyType S) {
    // CHECK-MESSAGES: [[@LINE-1]]:36: warning: the parameter 'S'
    // CHECK-FIXES: auto fn = [](const ExpensiveToCopyType& S) {
  };
  fn(ExpensiveToCopyType());
}

void lambdaConstAutoValue() {
  auto fn = [](const auto S) {
    // CHECK-MESSAGES: [[@LINE-1]]:27: warning: the const qualified parameter 'S'
    // CHECK-FIXES: auto fn = [](const auto& S) {
  };
  fn(ExpensiveToCopyType());
}

void lambdaNonConstAutoValue() {
  auto fn = [](auto S) {
    // CHECK-MESSAGES: [[@LINE-1]]:21: warning: the parameter 'S'
    // CHECK-FIXES: auto fn = [](const auto& S) {
  };
  fn(ExpensiveToCopyType());
}
