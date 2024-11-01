// RUN: %check_clang_tidy %s performance-trivially-destructible %t
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,performance-trivially-destructible' -fix --
// RUN: clang-tidy %t.cpp -checks='-*,performance-trivially-destructible' -warnings-as-errors='-*,performance-trivially-destructible' --

struct TriviallyDestructible1 {
  int a;
};

struct TriviallyDestructible2 : TriviallyDestructible1 {
  ~TriviallyDestructible2() = default;
  TriviallyDestructible1 b;
};

struct NotTriviallyDestructible1 : TriviallyDestructible2 {
  ~NotTriviallyDestructible1();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: class 'NotTriviallyDestructible1' can be made trivially destructible by defaulting the destructor on its first declaration [performance-trivially-destructible]
  // CHECK-FIXES: ~NotTriviallyDestructible1() = default;
  TriviallyDestructible2 b;
};

NotTriviallyDestructible1::~NotTriviallyDestructible1() = default; // to-be-removed
// CHECK-MESSAGES: :[[@LINE-1]]:28: note: destructor definition is here
// CHECK-FIXES: {{^}}// to-be-removed

// Don't emit for class template with type-dependent fields.
template <class T>
struct MaybeTriviallyDestructible1 {
  ~MaybeTriviallyDestructible1() noexcept;
  T t;
};

template <class T>
MaybeTriviallyDestructible1<T>::~MaybeTriviallyDestructible1() noexcept = default;

// Don't emit for specializations.
template struct MaybeTriviallyDestructible1<int>;

// Don't emit for class template with type-dependent bases.
template <class T>
struct MaybeTriviallyDestructible2 : T {
  ~MaybeTriviallyDestructible2() noexcept;
};

template <class T>
MaybeTriviallyDestructible2<T>::~MaybeTriviallyDestructible2() noexcept = default;

// Emit for templates without dependent bases and fields.
template <class T>
struct MaybeTriviallyDestructible1<T *> {
  ~MaybeTriviallyDestructible1() noexcept;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: class 'MaybeTriviallyDestructible1<T *>' can be made trivially destructible by defaulting the destructor on its first declaration [performance-trivially-destructible]
  // CHECK-FIXES: ~MaybeTriviallyDestructible1() noexcept = default;
  TriviallyDestructible1 t;
};

template <class T>
MaybeTriviallyDestructible1<T *>::~MaybeTriviallyDestructible1() noexcept = default; // to-be-removed
// CHECK-MESSAGES: :[[@LINE-1]]:35: note: destructor definition is here
// CHECK-FIXES: {{^}}// to-be-removed

// Emit for explicit specializations.
template <>
struct MaybeTriviallyDestructible1<double>: TriviallyDestructible1 {
  ~MaybeTriviallyDestructible1() noexcept;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: class 'MaybeTriviallyDestructible1<double>' can be made trivially destructible by defaulting the destructor on its first declaration [performance-trivially-destructible]
  // CHECK-FIXES: ~MaybeTriviallyDestructible1() noexcept = default;
};

MaybeTriviallyDestructible1<double>::~MaybeTriviallyDestructible1() noexcept = default; // to-be-removed
// CHECK-MESSAGES: :[[@LINE-1]]:38: note: destructor definition is here
// CHECK-FIXES: {{^}}// to-be-removed

struct NotTriviallyDestructible2 {
  virtual ~NotTriviallyDestructible2();
};

NotTriviallyDestructible2::~NotTriviallyDestructible2() = default;

struct NotTriviallyDestructible3: NotTriviallyDestructible2 {
  ~NotTriviallyDestructible3();
};

NotTriviallyDestructible3::~NotTriviallyDestructible3() = default;
