// RUN: %check_clang_tidy %s performance-implicit-cast-in-loop %t

// ---------- Classes used in the tests ----------

// Iterator returning by value.
template <typename T>
struct Iterator {
  void operator++();
  T operator*();
  bool operator!=(const Iterator& other);
};

// Iterator returning by reference.
template <typename T>
struct RefIterator {
  void operator++();
  T& operator*();
  bool operator!=(const RefIterator& other);
};

// The template argument is an iterator type, and a view is an object you can
// run a for loop on.
template <typename T>
struct View {
  T begin();
  T end();
};

// With this class, the implicit cast is a call to the (implicit) constructor of
// the class.
template <typename T>
class ImplicitWrapper {
 public:
  // Implicit!
  ImplicitWrapper(const T& t);
};

// With this class, the implicit cast is a call to the conversion operators of
// SimpleClass and ComplexClass.
template <typename T>
class OperatorWrapper {
 public:
  explicit OperatorWrapper(const T& t);
};

struct SimpleClass {
  int foo;
  operator OperatorWrapper<SimpleClass>();
};

// The materialize expression is not the same when the class has a destructor,
// so we make sure we cover that case too.
class ComplexClass {
 public:
  ComplexClass();
  ~ComplexClass();
  operator OperatorWrapper<ComplexClass>();
};

typedef View<Iterator<SimpleClass>> SimpleView;
typedef View<RefIterator<SimpleClass>> SimpleRefView;
typedef View<Iterator<ComplexClass>> ComplexView;
typedef View<RefIterator<ComplexClass>> ComplexRefView;

// ---------- The test themselves ----------
// For each test we do, in the same order, const ref, non const ref, const
// value, non const value.

void SimpleClassIterator() {
  for (const SimpleClass& foo : SimpleView()) {}
  // This line does not compile because a temporary cannot be assigned to a non
  // const reference.
  // for (SimpleClass& foo : SimpleView()) {}
  for (const SimpleClass foo : SimpleView()) {}
  for (SimpleClass foo : SimpleView()) {}
}

void SimpleClassRefIterator() {
  for (const SimpleClass& foo : SimpleRefView()) {}
  for (SimpleClass& foo : SimpleRefView()) {}
  for (const SimpleClass foo : SimpleRefView()) {}
  for (SimpleClass foo : SimpleRefView()) {}
}

void ComplexClassIterator() {
  for (const ComplexClass& foo : ComplexView()) {}
  // for (ComplexClass& foo : ComplexView()) {}
  for (const ComplexClass foo : ComplexView()) {}
  for (ComplexClass foo : ComplexView()) {}
}

void ComplexClassRefIterator() {
  for (const ComplexClass& foo : ComplexRefView()) {}
  for (ComplexClass& foo : ComplexRefView()) {}
  for (const ComplexClass foo : ComplexRefView()) {}
  for (ComplexClass foo : ComplexRefView()) {}
}

void ImplicitSimpleClassIterator() {
  for (const ImplicitWrapper<SimpleClass>& foo : SimpleView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the loop variable 'foo' is different from the one returned by the iterator and generates an implicit cast; you can either change the type to the correct one ('const SimpleClass &' but 'const auto&' is always a valid option) or remove the reference to make it explicit that you are creating a new value [performance-implicit-cast-in-loop]
  // for (ImplicitWrapper<SimpleClass>& foo : SimpleView()) {}
  for (const ImplicitWrapper<SimpleClass> foo : SimpleView()) {}
  for (ImplicitWrapper<SimpleClass>foo : SimpleView()) {}
}

void ImplicitSimpleClassRefIterator() {
  for (const ImplicitWrapper<SimpleClass>& foo : SimpleRefView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const SimpleClass &'.*}}
  // for (ImplicitWrapper<SimpleClass>& foo : SimpleRefView()) {}
  for (const ImplicitWrapper<SimpleClass> foo : SimpleRefView()) {}
  for (ImplicitWrapper<SimpleClass>foo : SimpleRefView()) {}
}

void ImplicitComplexClassIterator() {
  for (const ImplicitWrapper<ComplexClass>& foo : ComplexView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const ComplexClass &'.*}}
  // for (ImplicitWrapper<ComplexClass>& foo : ComplexView()) {}
  for (const ImplicitWrapper<ComplexClass> foo : ComplexView()) {}
  for (ImplicitWrapper<ComplexClass>foo : ComplexView()) {}
}

void ImplicitComplexClassRefIterator() {
  for (const ImplicitWrapper<ComplexClass>& foo : ComplexRefView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const ComplexClass &'.*}}
  // for (ImplicitWrapper<ComplexClass>& foo : ComplexRefView()) {}
  for (const ImplicitWrapper<ComplexClass> foo : ComplexRefView()) {}
  for (ImplicitWrapper<ComplexClass>foo : ComplexRefView()) {}
}

void OperatorSimpleClassIterator() {
  for (const OperatorWrapper<SimpleClass>& foo : SimpleView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const SimpleClass &'.*}}
  // for (OperatorWrapper<SimpleClass>& foo : SimpleView()) {}
  for (const OperatorWrapper<SimpleClass> foo : SimpleView()) {}
  for (OperatorWrapper<SimpleClass>foo : SimpleView()) {}
}

void OperatorSimpleClassRefIterator() {
  for (const OperatorWrapper<SimpleClass>& foo : SimpleRefView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const SimpleClass &'.*}}
  // for (OperatorWrapper<SimpleClass>& foo : SimpleRefView()) {}
  for (const OperatorWrapper<SimpleClass> foo : SimpleRefView()) {}
  for (OperatorWrapper<SimpleClass>foo : SimpleRefView()) {}
}

void OperatorComplexClassIterator() {
  for (const OperatorWrapper<ComplexClass>& foo : ComplexView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const ComplexClass &'.*}}
  // for (OperatorWrapper<ComplexClass>& foo : ComplexView()) {}
  for (const OperatorWrapper<ComplexClass> foo : ComplexView()) {}
  for (OperatorWrapper<ComplexClass>foo : ComplexView()) {}
}

void OperatorComplexClassRefIterator() {
  for (const OperatorWrapper<ComplexClass>& foo : ComplexRefView()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:{{[0-9]*}}: warning: the type of the{{.*'const ComplexClass &'.*}}
  // for (OperatorWrapper<ComplexClass>& foo : ComplexRefView()) {}
  for (const OperatorWrapper<ComplexClass> foo : ComplexRefView()) {}
  for (OperatorWrapper<ComplexClass>foo : ComplexRefView()) {}
}
