// RUN: %clang_cc1 -fsyntax-only -Wunused-private-field -Wused-but-marked-unused -Wno-uninitialized -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-private-field -Wused-but-marked-unused -Wno-uninitialized -verify -std=c++17 %s

class NotFullyDefined {
 public:
  NotFullyDefined();
 private:
  int y;
};

class HasUndefinedNestedClass {
  class Undefined;
  int unused_;
};

class HasUndefinedPureVirtualDestructor {
  virtual ~HasUndefinedPureVirtualDestructor() = 0;
  int unused_;
};

class HasDefinedNestedClasses {
  class DefinedHere {};
  class DefinedOutside;
  int unused_; // expected-warning{{private field 'unused_' is not used}}
};
class HasDefinedNestedClasses::DefinedOutside {};

class HasUndefinedFriendFunction {
  friend void undefinedFriendFunction();
  int unused_;
};

class HasUndefinedFriendClass {
  friend class NotFullyDefined;
  friend class NotDefined;
  int unused_;
};

class HasFriend {
  friend class FriendClass;
  friend void friendFunction(HasFriend f);
  int unused_; // expected-warning{{private field 'unused_' is not used}}
  int used_by_friend_class_;
  int used_by_friend_function_;
};

class ClassWithTemplateFriend {
  template <typename T> friend class TemplateFriend;
  int used_by_friend_;
  int unused_;
};

template <typename T> class TemplateFriend {
public:
  TemplateFriend(ClassWithTemplateFriend my_friend) {
    int var = my_friend.used_by_friend_;
  }
};

class FriendClass {
  HasFriend my_friend_;
  void use() {
    my_friend_.used_by_friend_class_ = 42;
  }
};

void friendFunction(HasFriend my_friend) {
  my_friend.used_by_friend_function_ = 42;
}

class NonTrivialConstructor {
 public:
  NonTrivialConstructor() {}
};

class NonTrivialDestructor {
 public:
  ~NonTrivialDestructor() {}
};

class Trivial {
 public:
  Trivial() = default;
  Trivial(int a) {}
};

int side_effect() {
  return 42;
}

class A {
 public:
  A() : primitive_type_(42), default_initializer_(), other_initializer_(42),
        trivial_(), user_constructor_(42),
        initialized_with_side_effect_(side_effect()) {
    used_ = 42;
    attr_used_ = 42; // expected-warning{{'attr_used_' was marked unused but was used}}
  }

  A(int x, A* a) : pointer_(a) {}

 private:
  int primitive_type_; // expected-warning{{private field 'primitive_type_' is not used}}
  A* pointer_; // expected-warning{{private field 'pointer_' is not used}}
  int no_initializer_; // expected-warning{{private field 'no_initializer_' is not used}}
  int default_initializer_; // expected-warning{{private field 'default_initializer_' is not used}}
  int other_initializer_; // expected-warning{{private field 'other_initializer_' is not used}}
  int used_, unused_; // expected-warning{{private field 'unused_' is not used}}
  int in_class_initializer_ = 42; // expected-warning{{private field 'in_class_initializer_' is not used}}
  int in_class_initializer_with_side_effect_ = side_effect();
  Trivial trivial_initializer_ = Trivial(); // expected-warning{{private field 'trivial_initializer_' is not used}}
  Trivial non_trivial_initializer_ = Trivial(42);
  int initialized_with_side_effect_;
  static int static_fields_are_ignored_;

  Trivial trivial_; // expected-warning{{private field 'trivial_' is not used}}
  Trivial user_constructor_;
  NonTrivialConstructor non_trivial_constructor_;
  NonTrivialDestructor non_trivial_destructor_;

  int attr_ __attribute__((unused));
  int attr_used_ __attribute__((unused));
};

class EverythingUsed {
 public:
  EverythingUsed() : as_array_index_(0), var_(by_initializer_) {
    var_ = sizeof(sizeof_);
    int *use = &by_reference_;
    int test[2];
    test[as_array_index_] = 42;
    int EverythingUsed::*ptr = &EverythingUsed::by_pointer_to_member_;
  }

  template<class T>
  void useStuff(T t) {
    by_template_function_ = 42;
  }

 private:
  int var_;
  int sizeof_;
  int by_reference_;
  int by_template_function_;
  int as_array_index_;
  int by_initializer_;
  int by_pointer_to_member_;
};

class HasFeatureTest {
#if __has_feature(attribute_unused_on_fields)
  int unused_; // expected-warning{{private field 'unused_' is not used}}
  int unused2_ __attribute__((unused)); // no-warning
#endif
};

namespace templates {
class B {
  template <typename T> void f(T t);
  int a;
};
}  // namespace templates

namespace mutual_friends {
// Undefined methods make mutual friends undefined.
class A {
  int a;
  friend class B;
  void doSomethingToAOrB();
};
class B {
  int b;
  friend class A;
};

// Undefined friends do not make a mutual friend undefined.
class C {
  int c;
  void doSomethingElse() {}
  friend class E;
  friend class D;
};
class D {
  int d; // expected-warning{{private field 'd' is not used}}
  friend class C;
};

// Undefined nested classes make mutual friends undefined.
class F {
  int f;
  class G;
  friend class H;
};
class H {
  int h;
  friend class F;
};
}  // namespace mutual_friends

namespace anonymous_structs_unions {
class A {
 private:
  // FIXME: Look at the DeclContext for anonymous structs/unions.
  union {
    int *Aligner;
    unsigned char Data[8];
  };
};
union S {
 private:
  int *Aligner;
  unsigned char Data[8];
};
}  // namespace anonymous_structs_unions

namespace pr13413 {
class A {
  A() : p_(__null), b_(false), a_(this), p2_(nullptr) {}
  void* p_;  // expected-warning{{private field 'p_' is not used}}
  bool b_;  // expected-warning{{private field 'b_' is not used}}
  A* a_;  // expected-warning{{private field 'a_' is not used}}
  void* p2_;  // expected-warning{{private field 'p2_' is not used}}
};
}

namespace pr13543 {
  void f(int);
  void f(char);
  struct S {
    S() : p(&f) {}
  private:
    void (*p)(int); // expected-warning{{private field 'p' is not used}}
  };

  struct A { int n; };
  struct B {
    B() : a(A()) {}
    B(char) {}
    B(int n) : a{n}, b{(f(n), 0)} {}
  private:
    A a = A(); // expected-warning{{private field 'a' is not used}}
    A b;
  };

  struct X { ~X(); };
  class C {
    X x[4]; // no-warning
  };
}

class implicit_special_member {
public:
  static implicit_special_member make() { return implicit_special_member(); }

private:
  int n; // expected-warning{{private field 'n' is not used}}
};

class defaulted_special_member {
public:
  defaulted_special_member(const defaulted_special_member&) = default;

private:
  int n; // expected-warning{{private field 'n' is not used}}
};
