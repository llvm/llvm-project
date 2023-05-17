// RUN: %check_clang_tidy %s performance-unnecessary-copy-initialization %t -- -config="{CheckOptions: [{key: performance-unnecessary-copy-initialization.ExcludedContainerTypes, value: 'ns::ViewType$;::ConstInCorrectType$'}]}" --

namespace ns {
template <typename T>
struct ViewType {
  ViewType(const T &);
  const T &view() const;
};
} // namespace ns

template <typename T>
struct ViewType {
  ViewType(const T &);
  const T &view() const;
};

struct ExpensiveToCopy {
  ~ExpensiveToCopy();
  void constMethod() const;
};

struct ConstInCorrectType {
  const ExpensiveToCopy &secretlyMutates() const;
  const ExpensiveToCopy &operator[](int) const;
};

using NSVTE = ns::ViewType<ExpensiveToCopy>;
typedef ns::ViewType<ExpensiveToCopy> FewType;

void positiveViewType() {
  ExpensiveToCopy E;
  ViewType<ExpensiveToCopy> V(E);
  const auto O = V.view();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'O' is copy-constructed
  // CHECK-FIXES: const auto& O = V.view();
  O.constMethod();
}

void excludedViewTypeInNamespace() {
  ExpensiveToCopy E;
  ns::ViewType<ExpensiveToCopy> V(E);
  const auto O = V.view();
  O.constMethod();
}

void excludedViewTypeAliased() {
  ExpensiveToCopy E;
  NSVTE V(E);
  const auto O = V.view();
  O.constMethod();

  FewType F(E);
  const auto P = F.view();
  P.constMethod();
}

void excludedConstIncorrectType() {
  ConstInCorrectType C;
  const auto E = C.secretlyMutates();
  E.constMethod();
}

void excludedConstIncorrectTypeOperator() {
  ConstInCorrectType C;
  const auto E = C[42];
  E.constMethod();
}

void excludedConstIncorrectTypeAsPointer(ConstInCorrectType *C) {
  const auto E = C->secretlyMutates();
  E.constMethod();
}

void excludedConstIncorrectTypeAsPointerOperator(ConstInCorrectType *C) {
  const auto E = (*C)[42];
  E.constMethod();
}

void excludedConstIncorrectTypeAsReference(const ConstInCorrectType &C) {
  const auto E = C.secretlyMutates();
  E.constMethod();
}

void excludedConstIncorrectTypeAsReferenceOperator(const ConstInCorrectType &C) {
  const auto E = C[42];
  E.constMethod();
}
