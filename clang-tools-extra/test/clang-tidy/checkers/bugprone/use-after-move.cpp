// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-use-after-move %t -- -- -fno-delayed-template-parsing

typedef decltype(nullptr) nullptr_t;

namespace std {
typedef unsigned size_t;

template <typename T>
struct unique_ptr {
  unique_ptr();
  T *get() const;
  explicit operator bool() const;
  void reset(T *ptr);
  T &operator*() const;
  T *operator->() const;
  T& operator[](size_t i) const;
};

template <typename T>
struct shared_ptr {
  shared_ptr();
  T *get() const;
  explicit operator bool() const;
  void reset(T *ptr);
  T &operator*() const;
  T *operator->() const;
};

template <typename T>
struct weak_ptr {
  weak_ptr();
  bool expired() const;
};

template <typename T1, typename T2>
struct pair {};

template <typename Key, typename T>
struct map {
  struct iterator {};

  map();
  void clear();
  bool empty();
  template <class... Args>
  pair<iterator, bool> try_emplace(const Key &key, Args &&...args);
};

template <typename Key, typename T>
struct unordered_map {
  struct iterator {};

  unordered_map();
  void clear();
  bool empty();
  template <class... Args>
  pair<iterator, bool> try_emplace(const Key &key, Args &&...args);
};

#define DECLARE_STANDARD_CONTAINER(name) \
  template <typename T>                  \
  struct name {                          \
    name();                              \
    void clear();                        \
    bool empty();                        \
  }

#define DECLARE_STANDARD_CONTAINER_WITH_ASSIGN(name) \
  template <typename T>                              \
  struct name {                                      \
    name();                                          \
    void clear();                                    \
    bool empty();                                    \
    void assign(size_t, const T &);                  \
  }

DECLARE_STANDARD_CONTAINER_WITH_ASSIGN(basic_string);
DECLARE_STANDARD_CONTAINER_WITH_ASSIGN(vector);
DECLARE_STANDARD_CONTAINER_WITH_ASSIGN(deque);
DECLARE_STANDARD_CONTAINER_WITH_ASSIGN(forward_list);
DECLARE_STANDARD_CONTAINER_WITH_ASSIGN(list);
DECLARE_STANDARD_CONTAINER(set);
DECLARE_STANDARD_CONTAINER(multiset);
DECLARE_STANDARD_CONTAINER(multimap);
DECLARE_STANDARD_CONTAINER(unordered_set);
DECLARE_STANDARD_CONTAINER(unordered_multiset);
DECLARE_STANDARD_CONTAINER(unordered_multimap);

typedef basic_string<char> string;

template <typename>
struct remove_reference;

template <typename _Tp>
struct remove_reference {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &> {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &&> {
  typedef _Tp type;
};

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) noexcept {
  return static_cast<typename remove_reference<_Tp>::type &&>(__t);
}

} // namespace std

class A {
public:
  A();
  A(const A &);
  A(A &&);

  A &operator=(const A &);
  A &operator=(A &&);

  void foo() const;
  int getInt() const;

  operator bool() const;

  int i;
};

template <class T>
class AnnotatedContainer {
public:
  AnnotatedContainer();

  void foo() const;
  [[clang::reinitializes]] void clear();
};

////////////////////////////////////////////////////////////////////////////////
// General tests.

// Simple case.
void simple() {
  A a;
  a.foo();
  A other_a = std::move(a);
  a.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:15: note: move occurred here
}

// Don't flag a move-to-self.
void selfMove() {
  A a;
  a = std::move(a);
  a.foo();
}

// A warning should only be emitted for one use-after-move.
void onlyFlagOneUseAfterMove() {
  A a;
  a.foo();
  A other_a = std::move(a);
  a.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:15: note: move occurred here
  a.foo();
}

void moveAfterMove() {
  // Move-after-move also counts as a use.
  {
    A a;
    std::move(a);
    std::move(a);
    // CHECK-NOTES: [[@LINE-1]]:15: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // This is also true if the move itself turns into the use on the second loop
  // iteration.
  {
    A a;
    for (int i = 0; i < 10; ++i) {
      std::move(a);
      // CHECK-NOTES: [[@LINE-1]]:17: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-2]]:7: note: move occurred here
      // CHECK-NOTES: [[@LINE-3]]:17: note: the use happens in a later loop
    }
  }
}

// Checks also works on function parameters that have a use-after move.
void parameters(A a) {
  std::move(a);
  a.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:3: note: move occurred here
}

void standardSmartPtr() {
  // std::unique_ptr<>, std::shared_ptr<> and std::weak_ptr<> are guaranteed to
  // be null after a std::move. So the check only flags accesses that would
  // dereference the pointer.
  {
    std::unique_ptr<A> ptr;
    std::move(ptr);
    ptr.get();
    static_cast<bool>(ptr);
    *ptr;
    // CHECK-NOTES: [[@LINE-1]]:6: warning: 'ptr' used after it was moved
    // CHECK-NOTES: [[@LINE-5]]:5: note: move occurred here
  }
  {
    std::unique_ptr<A> ptr;
    std::move(ptr);
    ptr->foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'ptr' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  {
    std::unique_ptr<A> ptr;
    std::move(ptr);
    ptr[0];
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'ptr' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  {
    std::shared_ptr<A> ptr;
    std::move(ptr);
    ptr.get();
    static_cast<bool>(ptr);
    *ptr;
    // CHECK-NOTES: [[@LINE-1]]:6: warning: 'ptr' used after it was moved
    // CHECK-NOTES: [[@LINE-5]]:5: note: move occurred here
  }
  {
    std::shared_ptr<A> ptr;
    std::move(ptr);
    ptr->foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'ptr' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  {
    // std::weak_ptr<> cannot be dereferenced directly, so we only check that
    // member functions may be called on it after a move.
    std::weak_ptr<A> ptr;
    std::move(ptr);
    ptr.expired();
  }
  // Make sure we recognize std::unique_ptr<> or std::shared_ptr<> if they're
  // wrapped in a typedef.
  {
    typedef std::unique_ptr<A> PtrToA;
    PtrToA ptr;
    std::move(ptr);
    ptr.get();
  }
  {
    typedef std::shared_ptr<A> PtrToA;
    PtrToA ptr;
    std::move(ptr);
    ptr.get();
  }
  // And we don't get confused if the template argument is a little more
  // involved.
  {
    struct B {
      typedef A AnotherNameForA;
    };
    std::unique_ptr<B::AnotherNameForA> ptr;
    std::move(ptr);
    ptr.get();
  }
  // Make sure we treat references to smart pointers correctly.
  {
    std::unique_ptr<A> ptr;
    std::unique_ptr<A>& ref_to_ptr = ptr;
    std::move(ref_to_ptr);
    ref_to_ptr.get();
  }
  {
    std::unique_ptr<A> ptr;
    std::unique_ptr<A>&& rvalue_ref_to_ptr = std::move(ptr);
    std::move(rvalue_ref_to_ptr);
    rvalue_ref_to_ptr.get();
  }
  // We don't give any special treatment to types that are called "unique_ptr"
  // or "shared_ptr" but are not in the "::std" namespace.
  {
    struct unique_ptr {
      void get();
    } ptr;
    std::move(ptr);
    ptr.get();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'ptr' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
}

// The check also works in member functions.
class Container {
  void useAfterMoveInMemberFunction() {
    A a;
    std::move(a);
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
};

// We see the std::move() if it's inside a declaration.
void moveInDeclaration() {
  A a;
  A another_a(std::move(a));
  a.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
}

// We see the std::move if it's inside an initializer list. Initializer lists
// are a special case because they cause ASTContext::getParents() to return
// multiple parents for certain nodes in their subtree. This is because
// RecursiveASTVisitor visits both the syntactic and semantic forms of
// InitListExpr, and the parent-child relationships are different between the
// two forms.
void moveInInitList() {
  struct S {
    A a;
  };
  A a;
  S s{std::move(a)};
  a.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:7: note: move occurred here
}

void lambdas() {
  // Use-after-moves inside a lambda should be detected.
  {
    A a;
    auto lambda = [a] {
      std::move(a);
      a.foo();
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-3]]:7: note: move occurred here
    };
  }
  // This is just as true if the variable was declared inside the lambda.
  {
    auto lambda = [] {
      A a;
      std::move(a);
      a.foo();
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-3]]:7: note: move occurred here
    };
  }
  // But don't warn if the move happened inside the lambda but the use happened
  // outside -- because
  // - the 'a' inside the lambda is a copy, and
  // - we don't know when the lambda will get called anyway
  {
    A a;
    auto lambda = [a] {
      std::move(a);
    };
    a.foo();
  }
  // Warn if the use consists of a capture that happens after a move.
  {
    A a;
    std::move(a);
    auto lambda = [a]() { a.foo(); };
    // CHECK-NOTES: [[@LINE-1]]:20: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // ...even if the capture was implicit.
  {
    A a;
    std::move(a);
    auto lambda = [=]() { a.foo(); };
    // CHECK-NOTES: [[@LINE-1]]:20: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // Same tests but for capture by reference.
  {
    A a;
    std::move(a);
    auto lambda = [&a]() { a.foo(); };
    // CHECK-NOTES: [[@LINE-1]]:21: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  {
    A a;
    std::move(a);
    auto lambda = [&]() { a.foo(); };
    // CHECK-NOTES: [[@LINE-1]]:20: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // But don't warn if the move happened after the capture.
  {
    A a;
    auto lambda = [a]() { a.foo(); };
    std::move(a);
  }
  // ...and again, same thing with an implicit move.
  {
    A a;
    auto lambda = [=]() { a.foo(); };
    std::move(a);
  }
  // Same tests but for capture by reference.
  {
    A a;
    auto lambda = [&a]() { a.foo(); };
    std::move(a);
  }
  {
    A a;
    auto lambda = [&]() { a.foo(); };
    std::move(a);
  }
  {
    A a;
    auto lambda = [a = std::move(a)] { a.foo(); };
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:24: note: move occurred here
  }
}

// Use-after-moves are detected in uninstantiated templates if the moved type
// is not a dependent type.
template <class T>
void movedTypeIsNotDependentType() {
  T t;
  A a;
  std::move(a);
  a.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:3: note: move occurred here
}

// And if the moved type is a dependent type, the use-after-move is detected if
// the template is instantiated.
template <class T>
void movedTypeIsDependentType() {
  T t;
  std::move(t);
  t.foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 't' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:3: note: move occurred here
}
template void movedTypeIsDependentType<A>();

// We handle the case correctly where the move consists of an implicit call
// to a conversion operator.
void implicitConversionOperator() {
  struct Convertible {
    operator A() && { return A(); }
  };
  void takeA(A a);

  Convertible convertible;
  takeA(std::move(convertible));
  convertible;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: 'convertible' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:9: note: move occurred here
}

// Using decltype on an expression is not a use.
void decltypeIsNotUse() {
  A a;
  std::move(a);
  decltype(a) other_a;
}

// Ignore moves or uses that occur as part of template arguments.
template <int>
class ClassTemplate {
public:
  void foo(A a);
};
template <int>
void functionTemplate(A a);
void templateArgIsNotUse() {
  {
    // A pattern like this occurs in the EXPECT_EQ and ASSERT_EQ macros in
    // Google Test.
    A a;
    ClassTemplate<sizeof(A(std::move(a)))>().foo(std::move(a));
  }
  {
    A a;
    functionTemplate<sizeof(A(std::move(a)))>(std::move(a));
  }
}

// Ignore moves of global variables.
A global_a;
void ignoreGlobalVariables() {
  std::move(global_a);
  global_a.foo();
}

// Ignore moves of member variables.
class IgnoreMemberVariables {
  A a;
  static A static_a;

  void f() {
    std::move(a);
    a.foo();

    std::move(static_a);
    static_a.foo();
  }
};

// Ignore moves that happen in a try_emplace.
void ignoreMoveInTryEmplace() {
  {
    std::map<int, A> amap;
    A a;
    amap.try_emplace(1, std::move(a));
    a.foo();
  }
  {
    std::unordered_map<int, A> amap;
    A a;
    amap.try_emplace(1, std::move(a));
    a.foo();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Tests involving control flow.

void useAndMoveInLoop() {
  // Warn about use-after-moves if they happen in a later loop iteration than
  // the std::move().
  {
    A a;
    for (int i = 0; i < 10; ++i) {
      a.foo();
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE+2]]:7: note: move occurred here
      // CHECK-NOTES: [[@LINE-3]]:7: note: the use happens in a later loop
      std::move(a);
    }
  }
  // However, this case shouldn't be flagged -- the scope of the declaration of
  // 'a' is important.
  {
    for (int i = 0; i < 10; ++i) {
      A a;
      a.foo();
      std::move(a);
    }
  }
  // Same as above, except that we have an unrelated variable being declared in
  // the same declaration as 'a'. This case is interesting because it tests that
  // the synthetic DeclStmts generated by the CFG are sequenced correctly
  // relative to the other statements.
  {
    for (int i = 0; i < 10; ++i) {
      A a, other;
      a.foo();
      std::move(a);
    }
  }
  // Don't warn if we return after the move.
  {
    A a;
    for (int i = 0; i < 10; ++i) {
      a.foo();
      if (a.getInt() > 0) {
        std::move(a);
        return;
      }
    }
  }
}

void differentBranches(int i) {
  // Don't warn if the use is in a different branch from the move.
  {
    A a;
    if (i > 0) {
      std::move(a);
    } else {
      a.foo();
    }
  }
  // Same thing, but with a ternary operator.
  {
    A a;
    i > 0 ? (void)std::move(a) : a.foo();
  }
  // A variation on the theme above.
  {
    A a;
    a.getInt() > 0 ? a.getInt() : A(std::move(a)).getInt();
  }
  // Same thing, but with a switch statement.
  {
    A a;
    switch (i) {
    case 1:
      std::move(a);
      break;
    case 2:
      a.foo();
      break;
    }
  }
  // However, if there's a fallthrough, we do warn.
  {
    A a;
    switch (i) {
    case 1:
      std::move(a);
    case 2:
      a.foo();
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-4]]:7: note: move occurred here
      break;
    }
  }
}

// False positive: A use-after-move is flagged even though the "if (b)" and
// "if (!b)" are mutually exclusive.
void mutuallyExclusiveBranchesFalsePositive(bool b) {
  A a;
  if (b) {
    std::move(a);
  }
  if (!b) {
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-5]]:5: note: move occurred here
  }
}

// Destructors marked [[noreturn]] are handled correctly in the control flow
// analysis. (These are used in some styles of assertion macros.)
class FailureLogger {
public:
  FailureLogger();
  [[noreturn]] ~FailureLogger();
  void log(const char *);
};
#define ASSERT(x) \
  while (x)       \
  FailureLogger().log(#x)
bool operationOnA(A);
void noreturnDestructor() {
  A a;
  // The while loop in the ASSERT() would ordinarily have the potential to cause
  // a use-after-move because the second iteration of the loop would be using a
  // variable that had been moved from in the first iteration. Check that the
  // CFG knows that the second iteration of the loop is never reached because
  // the FailureLogger destructor is marked [[noreturn]].
  ASSERT(operationOnA(std::move(a)));
}
#undef ASSERT

////////////////////////////////////////////////////////////////////////////////
// Tests for reinitializations

template <class T>
void swap(T &a, T &b) {
  T tmp = std::move(a);
  a = std::move(b);
  b = std::move(tmp);
}
void assignments(int i) {
  // Don't report a use-after-move if the variable was assigned to in the
  // meantime.
  {
    A a;
    std::move(a);
    a = A();
    a.foo();
  }
  // The assignment should also be recognized if move, assignment and use don't
  // all happen in the same block (but the assignment is still guaranteed to
  // prevent a use-after-move).
  {
    A a;
    if (i == 1) {
      std::move(a);
      a = A();
    }
    if (i == 2) {
      a.foo();
    }
  }
  {
    A a;
    if (i == 1) {
      std::move(a);
    }
    if (i == 2) {
      a = A();
      a.foo();
    }
  }
  // The built-in assignment operator should also be recognized as a
  // reinitialization. (std::move() may be called on built-in types in template
  // code.)
  {
    int a1 = 1, a2 = 2;
    swap(a1, a2);
  }
  // A std::move() after the assignment makes the variable invalid again.
  {
    A a;
    std::move(a);
    a = A();
    std::move(a);
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // Report a use-after-move if we can't be sure that the variable was assigned
  // to.
  {
    A a;
    std::move(a);
    if (i < 10) {
      a = A();
    }
    if (i > 5) {
      a.foo();
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-7]]:5: note: move occurred here
    }
  }
}

// Passing the object to a function through a non-const pointer or reference
// counts as a re-initialization.
void passByNonConstPointer(A *);
void passByNonConstReference(A &);
void passByNonConstPointerIsReinit() {
  {
    A a;
    std::move(a);
    passByNonConstPointer(&a);
    a.foo();
  }
  {
    A a;
    std::move(a);
    passByNonConstReference(a);
    a.foo();
  }
}

// Passing the object through a const pointer or reference counts as a use --
// since the called function cannot reinitialize the object.
void passByConstPointer(const A *);
void passByConstReference(const A &);
void passByConstPointerIsUse() {
  {
    // Declaring 'a' as const so that no ImplicitCastExpr is inserted into the
    // AST -- we wouldn't want the check to rely solely on that to detect a
    // const pointer argument.
    const A a;
    std::move(a);
    passByConstPointer(&a);
    // CHECK-NOTES: [[@LINE-1]]:25: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  const A a;
  std::move(a);
  passByConstReference(a);
  // CHECK-NOTES: [[@LINE-1]]:24: warning: 'a' used after it was moved
  // CHECK-NOTES: [[@LINE-3]]:3: note: move occurred here
}

// Clearing a standard container using clear() is treated as a
// re-initialization.
void standardContainerClearIsReinit() {
  {
    std::string container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::vector<int> container;
    std::move(container);
    container.clear();
    container.empty();

    auto container2 = container;
    std::move(container2);
    container2.clear();
    container2.empty();
  }
  {
    std::deque<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::forward_list<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::list<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::set<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::map<int, int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::multiset<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::multimap<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::unordered_set<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::unordered_map<int, int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::unordered_multiset<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  {
    std::unordered_multimap<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  // This should also work for typedefs of standard containers.
  {
    typedef std::vector<int> IntVector;
    IntVector container;
    std::move(container);
    container.clear();
    container.empty();
  }
  // But it shouldn't work for non-standard containers.
  {
    // This might be called "vector", but it's not in namespace "std".
    struct vector {
      void clear() {}
    } container;
    std::move(container);
    container.clear();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'container' used after it was
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // An intervening clear() on a different container does not reinitialize.
  {
    std::vector<int> container1, container2;
    std::move(container1);
    container2.clear();
    container1.empty();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'container1' used after it was
    // CHECK-NOTES: [[@LINE-4]]:5: note: move occurred here
  }
}

// Clearing a standard container using assign() is treated as a
// re-initialization.
void standardContainerAssignIsReinit() {
  {
    std::string container;
    std::move(container);
    container.assign(0, ' ');
    container.empty();
  }
  {
    std::vector<int> container;
    std::move(container);
    container.assign(0, 0);
    container.empty();
  }
  {
    std::deque<int> container;
    std::move(container);
    container.assign(0, 0);
    container.empty();
  }
  {
    std::forward_list<int> container;
    std::move(container);
    container.assign(0, 0);
    container.empty();
  }
  {
    std::list<int> container;
    std::move(container);
    container.clear();
    container.empty();
  }
  // But it doesn't work for non-standard containers.
  {
    // This might be called "vector", but it's not in namespace "std".
    struct vector {
      void assign(std::size_t, int) {}
    } container;
    std::move(container);
    container.assign(0, 0);
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'container' used after it was
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  // An intervening assign() on a different container does not reinitialize.
  {
    std::vector<int> container1, container2;
    std::move(container1);
    container2.assign(0, 0);
    container1.empty();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'container1' used after it was
    // CHECK-NOTES: [[@LINE-4]]:5: note: move occurred here
  }
}

// Resetting the standard smart pointer types using reset() is treated as a
// re-initialization. (We don't test std::weak_ptr<> because it can't be
// dereferenced directly.)
void standardSmartPointerResetIsReinit() {
  {
    std::unique_ptr<A> ptr;
    std::move(ptr);
    ptr.reset(new A);
    *ptr;
  }
  {
    std::shared_ptr<A> ptr;
    std::move(ptr);
    ptr.reset(new A);
    *ptr;
  }
}

void reinitAnnotation() {
  {
    AnnotatedContainer<int> obj;
    std::move(obj);
    obj.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'obj' used after it was
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  {
    AnnotatedContainer<int> obj;
    std::move(obj);
    obj.clear();
    obj.foo();
  }
  {
    // Calling clear() on a different object to the one that was moved is not
    // considered a reinitialization.
    AnnotatedContainer<int> obj1, obj2;
    std::move(obj1);
    obj2.clear();
    obj1.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'obj1' used after it was
    // CHECK-NOTES: [[@LINE-4]]:5: note: move occurred here
  }
}

////////////////////////////////////////////////////////////////////////////////
// Tests related to order of evaluation within expressions

// Relative sequencing of move and use.
void passByRvalueReference(int i, A &&a);
void passByValue(int i, A a);
void passByValue(A a, int i);
A g(A, A &&);
int intFromA(A &&);
int intFromInt(int);
void sequencingOfMoveAndUse() {
  // This case is fine because the move only happens inside
  // passByRvalueReference(). At this point, a.getInt() is guaranteed to have
  // been evaluated.
  {
    A a;
    passByRvalueReference(a.getInt(), std::move(a));
  }
  // However, if we pass by value, the move happens when the move constructor is
  // called to create a temporary, and this happens before the call to
  // passByValue(). Because the order in which arguments are evaluated isn't
  // defined, the move may happen before the call to a.getInt().
  //
  // Check that we warn about a potential use-after move for both orderings of
  // a.getInt() and std::move(a), independent of the order in which the
  // arguments happen to get evaluated by the compiler.
  {
    A a;
    passByValue(a.getInt(), std::move(a));
    // CHECK-NOTES: [[@LINE-1]]:17: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:29: note: move occurred here
    // CHECK-NOTES: [[@LINE-3]]:17: note: the use and move are unsequenced
  }
  {
    A a;
    passByValue(std::move(a), a.getInt());
    // CHECK-NOTES: [[@LINE-1]]:31: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:17: note: move occurred here
    // CHECK-NOTES: [[@LINE-3]]:31: note: the use and move are unsequenced
  }
  // An even more convoluted example.
  {
    A a;
    g(g(a, std::move(a)), g(a, std::move(a)));
    // CHECK-NOTES: [[@LINE-1]]:9: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:27: note: move occurred here
    // CHECK-NOTES: [[@LINE-3]]:9: note: the use and move are unsequenced
    // CHECK-NOTES: [[@LINE-4]]:29: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-5]]:7: note: move occurred here
    // CHECK-NOTES: [[@LINE-6]]:29: note: the use and move are unsequenced
  }
  // This case is fine because the actual move only happens inside the call to
  // operator=(). a.getInt(), by necessity, is evaluated before that call.
  {
    A a;
    A vec[1];
    vec[a.getInt()] = std::move(a);
  }
  // However, in the following case, the move happens before the assignment, and
  // so the order of evaluation is not guaranteed.
  {
    A a;
    int v[3];
    v[a.getInt()] = intFromA(std::move(a));
    // CHECK-NOTES: [[@LINE-1]]:7: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:21: note: move occurred here
    // CHECK-NOTES: [[@LINE-3]]:7: note: the use and move are unsequenced
  }
  {
    A a;
    int v[3];
    v[intFromA(std::move(a))] = intFromInt(a.i);
    // CHECK-NOTES: [[@LINE-1]]:44: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:7: note: move occurred here
    // CHECK-NOTES: [[@LINE-3]]:44: note: the use and move are unsequenced
  }
}

// Relative sequencing of move and reinitialization. If the two are unsequenced,
// we conservatively assume that the move happens after the reinitialization,
// i.e. the that object does not get reinitialized after the move.
A MutateA(A a);
void passByValue(A a1, A a2);
void sequencingOfMoveAndReinit() {
  // Move and reinitialization as function arguments (which are indeterminately
  // sequenced). Again, check that we warn for both orderings.
  {
    A a;
    passByValue(std::move(a), (a = A()));
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:17: note: move occurred here
  }
  {
    A a;
    passByValue((a = A()), std::move(a));
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:28: note: move occurred here
  }
  // Common usage pattern: Move the object to a function that mutates it in some
  // way, then reassign the result to the object. This pattern is fine.
  {
    A a;
    a = MutateA(std::move(a));
    a.foo();
  }
}

// Relative sequencing of reinitialization and use. If the two are unsequenced,
// we conservatively assume that the reinitialization happens after the use,
// i.e. that the object is not reinitialized at the point in time when it is
// used.
void sequencingOfReinitAndUse() {
  // Reinitialization and use in function arguments. Again, check both possible
  // orderings.
  {
    A a;
    std::move(a);
    passByValue(a.getInt(), (a = A()));
    // CHECK-NOTES: [[@LINE-1]]:17: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
  {
    A a;
    std::move(a);
    passByValue((a = A()), a.getInt());
    // CHECK-NOTES: [[@LINE-1]]:28: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:5: note: move occurred here
  }
}

// The comma operator sequences its operands.
void commaOperatorSequences() {
  {
    A a;
    A(std::move(a))
    , (a = A());
    a.foo();
  }
  {
    A a;
    (a = A()), A(std::move(a));
    a.foo();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:16: note: move occurred here
  }
}

// An initializer list sequences its initialization clauses.
void initializerListSequences() {
  {
    struct S1 {
      int i;
      A a;
    };
    A a;
    S1 s1{a.getInt(), std::move(a)};
  }
  {
    struct S2 {
      A a;
      int i;
    };
    A a;
    S2 s2{std::move(a), a.getInt()};
    // CHECK-NOTES: [[@LINE-1]]:25: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:11: note: move occurred here
  }
}

// A declaration statement containing multiple declarations sequences the
// initializer expressions.
void declarationSequences() {
  {
    A a;
    A a1 = a, a2 = std::move(a);
  }
  {
    A a;
    A a1 = std::move(a), a2 = a;
    // CHECK-NOTES: [[@LINE-1]]:31: warning: 'a' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:12: note: move occurred here
  }
}

// The logical operators && and || sequence their operands.
void logicalOperatorsSequence() {
  {
    A a;
    if (a.getInt() > 0 && A(std::move(a)).getInt() > 0) {
      A().foo();
    }
  }
  // A variation: Negate the result of the && (which pushes the && further down
  // into the AST).
  {
    A a;
    if (!(a.getInt() > 0 && A(std::move(a)).getInt() > 0)) {
      A().foo();
    }
  }
  {
    A a;
    if (A(std::move(a)).getInt() > 0 && a.getInt() > 0) {
      // CHECK-NOTES: [[@LINE-1]]:41: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-2]]:9: note: move occurred here
      A().foo();
    }
  }
  {
    A a;
    if (a.getInt() > 0 || A(std::move(a)).getInt() > 0) {
      A().foo();
    }
  }
  {
    A a;
    if (A(std::move(a)).getInt() > 0 || a.getInt() > 0) {
      // CHECK-NOTES: [[@LINE-1]]:41: warning: 'a' used after it was moved
      // CHECK-NOTES: [[@LINE-2]]:9: note: move occurred here
      A().foo();
    }
  }
}

// A range-based for sequences the loop variable declaration before the body.
void forRangeSequences() {
  A v[2] = {A(), A()};
  for (A &a : v) {
    std::move(a);
  }
}

// If a variable is declared in an if, while or switch statement, the init
// statement (for if and switch) is sequenced before the variable declaration,
// which in turn is sequenced before the evaluation of the condition. We place
// all tests inside a for loop to ensure that the checker understands the
// sequencing. If it didn't, then the loop would trigger the "moved twice"
// logic.
void ifWhileAndSwitchSequenceInitDeclAndCondition() {
  for (int i = 0; i < 10; ++i) {
    A a1;
    if (A a2 = std::move(a1)) {
      std::move(a2);
    }
  }
  for (int i = 0; i < 10; ++i) {
    A a1;
    if (A a2 = std::move(a1); a2) {
      std::move(a2);
    }
  }
  for (int i = 0; i < 10; ++i) {
    A a1;
    if (A a2 = std::move(a1); A a3 = std::move(a2)) {
      std::move(a3);
    }
  }
  for (int i = 0; i < 10; ++i) {
    // init followed by condition with move, but without variable declaration.
    if (A a1; A(std::move(a1)).getInt() > 0) {}
  }
  for (int i = 0; i < 10; ++i) {
    if (A a1; A(std::move(a1)).getInt() > a1.getInt()) {}
    // CHECK-NOTES: [[@LINE-1]]:43: warning: 'a1' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:15: note: move occurred here
    // CHECK-NOTES: [[@LINE-3]]:43: note: the use and move are unsequenced
  }
  for (int i = 0; i < 10; ++i) {
    A a1;
    if (A a2 = std::move(a1); A(a1) > 0) {}
    // CHECK-NOTES: [[@LINE-1]]:33: warning: 'a1' used after it was moved
    // CHECK-NOTES: [[@LINE-2]]:16: note: move occurred here
  }
  while (A a = A()) {
    std::move(a);
  }
  for (int i = 0; i < 10; ++i) {
    A a1;
    switch (A a2 = std::move(a1); a2) {
      case true:
        std::move(a2);
    }
  }
  for (int i = 0; i < 10; ++i) {
    A a1;
    switch (A a2 = a1; A a3 = std::move(a2)) {
      case true:
        std::move(a3);
    }
  }
}

// Some statements in templates (e.g. null, break and continue statements) may
// be shared between the uninstantiated and instantiated versions of the
// template and therefore have multiple parents. Make sure the sequencing code
// handles this correctly.
template <class> void nullStatementSequencesInTemplate() {
  int c = 0;
  (void)c;
  ;
  std::move(c);
}
template void nullStatementSequencesInTemplate<int>();

namespace PR33020 {
class D {
  ~D();
};
struct A {
  D d;
};
class B {
  A a;
};
template <typename T>
class C : T, B {
  void m_fn1() {
    int a;
    std::move(a);
    C c;
  }
};
} // namespace PR33020

namespace UnevalContext {
struct Foo {};
void noExcept() {
  Foo Bar;
  (void) noexcept(Foo{std::move(Bar)});
  Foo Other{std::move(Bar)};
}
void sizeOf() {
  Foo Bar;
  (void)sizeof(Foo{std::move(Bar)});
  Foo Other{std::move(Bar)};
}
void alignOf() {
  Foo Bar;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-alignof-expression"
  (void)alignof(Foo{std::move(Bar)});
#pragma clang diagnostic pop
  Foo Other{std::move(Bar)};
}
void typeId() {
  Foo Bar;
  // error: you need to include <typeinfo> before using the 'typeid' operator
  // (void) typeid(Foo{std::move(Bar)}).name();
  Foo Other{std::move(Bar)};
}
} // namespace UnevalContext

class PR38187 {
public:
  PR38187(std::string val) : val_(std::move(val)) {
    val.empty();
    // CHECK-NOTES: [[@LINE-1]]:5: warning: 'val' used after it was moved
    // CHECK-NOTES: [[@LINE-3]]:30: note: move occurred here
  }

private:
  std::string val_;
};
