// RUN: %check_clang_tidy %s modernize-use-emplace %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {modernize-use-emplace.PushLikeFunctions: \
// RUN:                '::Vector::append=emplaceBack; \
// RUN:                 Vector::infallibleAppend=infallibleEmplaceBack; \
// RUN:                 ArrayBase::AppendElement=EmplaceBack; \
// RUN:                 NoValueType::append=emplaceBack; \
// RUN:                 invalid'}}"

// CHECK-MESSAGES: warning: invalid entry 'invalid' in option 'modernize-use-emplace.PushLikeFunctions'; expected 'function=replacement' [clang-tidy-config]

template <typename T>
class Vector {
public:
  using value_type = T;
  template <typename U> bool append(U &&);
  template <typename... Args> bool emplaceBack(Args &&...);
  template <typename U> void infallibleAppend(U &&);
  template <typename... Args> void infallibleEmplaceBack(Args &&...);
  template <typename U> bool appendOther(U &&);
};

template <typename E>
class ArrayBase {
public:
  typedef E value_type;
  template <typename Item> E *AppendElement(Item &&);
  template <typename Item> E *AppendElement(Item &&, int);
  template <typename... Args> E *EmplaceBack(Args &&...);
};

template <typename E>
class Array : public ArrayBase<E> {
public:
  using typename ArrayBase<E>::value_type;
  using ArrayBase<E>::AppendElement;
};

template <typename T>
class NoValueType {
public:
  template <typename U> bool append(U &&);
  template <typename... Args> bool emplaceBack(Args &&...);
};

struct Something {
  Something(int A, int B = 41) {}
  Something() {}
};

struct Wrapper {
  Wrapper(Something) {}
};

void testVector() {
  Vector<Something> V;
  V.append(Something(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplaceBack instead of append [modernize-use-emplace]
  // CHECK-FIXES: V.emplaceBack(1, 2);

  V.append(Something{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplaceBack instead of append
  // CHECK-FIXES: V.emplaceBack(1, 2);

  V.append(Something());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplaceBack instead of append
  // CHECK-FIXES: V.emplaceBack();

  V.infallibleAppend(Something(3));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use infallibleEmplaceBack instead of infallibleAppend
  // CHECK-FIXES: V.infallibleEmplaceBack(3);

  V.appendOther(Something(1, 2));

  Something S;
  V.append(S);

  Vector<Wrapper> W;
  W.append(Something(1, 2));
}

void testArray(Array<Something> *P) {
  Array<Something> A;
  A.AppendElement(Something(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use EmplaceBack instead of AppendElement
  // CHECK-FIXES: A.EmplaceBack(1, 2);

  P->AppendElement(Something(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use EmplaceBack instead of AppendElement
  // CHECK-FIXES: P->EmplaceBack(1, 2);

  A.AppendElement(Something(1, 2), 0);
}

void testNoValueType() {
  NoValueType<Something> N;
  N.append(Something(1, 2));
}

template <typename T>
void testDependent() {
  Vector<T> V;
  V.append(T(1, 2));
}

void callDependent() { testDependent<Something>(); }
