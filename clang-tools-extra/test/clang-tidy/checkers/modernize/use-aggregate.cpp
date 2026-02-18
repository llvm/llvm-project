// RUN: %check_clang_tidy %s modernize-use-aggregate %t

// Positive: simple forwarding constructor.
struct Point {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'Point' can be an aggregate type if the forwarding constructor is removed [modernize-use-aggregate]
  int X;
  int Y;
  Point(int X, int Y) : X(X), Y(Y) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: remove this constructor to enable aggregate initialization
};

// Positive: forwarding constructor with class-type member (copy/move).
namespace std {
template <typename T>
class basic_string {
public:
  basic_string();
  basic_string(const basic_string &);
  basic_string(basic_string &&);
  basic_string(const char *);
};
using string = basic_string<char>;
} // namespace std

struct Person {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'Person' can be an aggregate type if the forwarding constructor is removed [modernize-use-aggregate]
  std::string Name;
  int Age;
  Person(std::string Name, int Age) : Name(Name), Age(Age) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: remove this constructor to enable aggregate initialization
};

// Negative: constructor does more than forward.
struct WithLogic {
  int X;
  int Y;
  WithLogic(int X, int Y) : X(X), Y(Y + 1) {}
};

// Negative: constructor body is not empty.
struct WithBody {
  int X;
  int Y;
  WithBody(int X, int Y) : X(X), Y(Y) { X++; }
};

// Negative: has virtual functions.
struct WithVirtual {
  int X;
  virtual void foo();
  WithVirtual(int X) : X(X) {}
};

// Negative: has private data members.
class WithPrivate {
  int X;
public:
  WithPrivate(int X) : X(X) {}
};

// Negative: has protected data members.
struct WithProtected {
protected:
  int X;
public:
  WithProtected(int X) : X(X) {}
};

// Negative: wrong parameter count.
struct WrongParamCount {
  int X;
  int Y;
  WrongParamCount(int X) : X(X), Y(0) {}
};

// Negative: wrong init order (param 1 -> field 0, param 0 -> field 1).
struct WrongOrder {
  int X;
  int Y;
  WrongOrder(int A, int B) : X(B), Y(A) {}
};

// Negative: has non-trivial destructor.
struct WithDestructor {
  int X;
  WithDestructor(int X) : X(X) {}
  ~WithDestructor() {}
};

// Negative: has additional non-trivial constructor.
struct MultiCtor {
  int X;
  MultiCtor(int X) : X(X) {}
  MultiCtor(double) : X(0) {}
};

// Negative: empty struct (already an aggregate).
struct Empty {
  Empty() {}
};

// Negative: template specialization.
template <typename T>
struct Templated {
  T X;
  Templated(T X) : X(X) {}
};
Templated<int> TI(1);

// Negative: virtual base class.
struct Base {};
struct WithVirtualBase : virtual Base {
  int X;
  WithVirtualBase(int X) : X(X) {}
};

// Negative: private base class.
struct WithPrivateBase : private Base {
  int X;
  WithPrivateBase(int X) : X(X) {}
};

// Positive: three fields.
struct Triple {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'Triple' can be an aggregate type if the forwarding constructor is removed [modernize-use-aggregate]
  int A;
  int B;
  int C;
  Triple(int A, int B, int C) : A(A), B(B), C(C) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: remove this constructor to enable aggregate initialization
};

// Positive: single field.
struct Single {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'Single' can be an aggregate type if the forwarding constructor is removed [modernize-use-aggregate]
  int X;
  Single(int X) : X(X) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: remove this constructor to enable aggregate initialization
};

// Negative: has base class initializer.
struct Derived : Base {
  int X;
  Derived(int X) : Base(), X(X) {}
};
