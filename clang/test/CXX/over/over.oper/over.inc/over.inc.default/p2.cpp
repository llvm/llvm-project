// RUN: %clang_cc1 -std=c++2d -verify %s

// C++2d [over.inc.default]p2:
//   A defaulted postfix increment or decrement operator function for a type C
//   is defined as deleted if
//    -- C is a class type for which overload resolution, as applied to
//       direct-initialization of a variable of type C from an lvalue of type
//       C, does not result in a usable candidate,
//    -- C has a destructor that is deleted or inaccessible from the context of
//       the function-body, or
//    -- for an lvalue c of type C, overload resolution as applied to ++c for a
//       postfix increment operator function or --c for a postfix decrement
//       operator function does not result in a usable candidate.

struct NoPrefix {
  NoPrefix operator++(int) = default; // #NoPrefix
  // expected-warning@#NoPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#NoPrefix 2 {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'NoPrefix'}}
  // expected-note@#NoPrefix {{replace 'default' with 'delete'}}
  // expected-note@#NoPrefix {{explicitly defaulted function was implicitly deleted here}}
};
void use_no_prefix(NoPrefix n) {
  n++; // expected-error {{object of type 'NoPrefix' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}

struct NoPrefixDecrement {
  NoPrefixDecrement &operator++();
  NoPrefixDecrement operator--(int) = default; // #NoPrefixDecrement
  // expected-warning@#NoPrefixDecrement {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#NoPrefixDecrement 2 {{defaulted 'operator--' is implicitly deleted because there is no viable prefix 'operator--' for an lvalue of type 'NoPrefixDecrement'}}
  // expected-note@#NoPrefixDecrement {{replace 'default' with 'delete'}}
  // expected-note@#NoPrefixDecrement {{explicitly defaulted function was implicitly deleted here}}
};
void use_no_prefix_decrement(NoPrefixDecrement n) {
  n--; // expected-error {{object of type 'NoPrefixDecrement' cannot be decremented because its defaulted postfix decrement operator is implicitly deleted}}
}

// There is no built-in increment operator for enumeration types.
enum class E {};
E operator++(E &, int) = default; // #E
// expected-warning@#E {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#E 2 {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'E'}}
// expected-note@#E {{replace 'default' with 'delete'}}
// expected-note@#E {{explicitly defaulted function was implicitly deleted here}}
void use_e(E e) {
  e++; // expected-error {{object of type 'E' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}

// A conversion to a built-in type provides a usable candidate.
struct ConvertsToInt {
  operator int &();
  ConvertsToInt operator++(int) = default; // OK
};
struct AmbiguousBuiltin {
  operator int &();
  operator long &();
  AmbiguousBuiltin operator++(int) = default; // #AmbiguousBuiltin
  // expected-warning@#AmbiguousBuiltin {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#AmbiguousBuiltin {{defaulted 'operator++' is implicitly deleted because prefix 'operator++' for an lvalue of type 'AmbiguousBuiltin' is ambiguous}}
  // expected-note@#AmbiguousBuiltin {{built-in candidate operator++(int &)}}
  // expected-note@#AmbiguousBuiltin {{built-in candidate operator++(long &)}}
  // expected-note@#AmbiguousBuiltin {{replace 'default' with 'delete'}}
};

struct DeletedPrefix {
  DeletedPrefix &operator++() = delete; // expected-note {{'operator++' has been explicitly marked deleted here}}
  DeletedPrefix operator++(int) = default; // #DeletedPrefix
  // expected-warning@#DeletedPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#DeletedPrefix {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted prefix 'operator++'}}
  // expected-note@#DeletedPrefix {{replace 'default' with 'delete'}}
};

struct RvaluePrefix {
  RvaluePrefix &operator++() &&;
  RvaluePrefix operator++(int) = default; // #RvaluePrefix
  // expected-warning@#RvaluePrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#RvaluePrefix {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'RvaluePrefix'}}
  // expected-note@#RvaluePrefix {{replace 'default' with 'delete'}}
};

class PrivatePrefix {
  PrivatePrefix &operator++(); // expected-note {{declared private here}}

public:
  PrivatePrefix operator--(int) = default; // #PrivatePrefixDecrement
  // expected-warning@#PrivatePrefixDecrement {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#PrivatePrefixDecrement {{defaulted 'operator--' is implicitly deleted because there is no viable prefix 'operator--' for an lvalue of type 'PrivatePrefix'}}
  // expected-note@#PrivatePrefixDecrement {{replace 'default' with 'delete'}}
};
PrivatePrefix operator++(PrivatePrefix &, int) = default; // #PrivatePrefix
// expected-warning@#PrivatePrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#PrivatePrefix {{defaulted 'operator++' is implicitly deleted because it would invoke a private prefix 'operator++' of 'PrivatePrefix'}}
// expected-note@#PrivatePrefix {{replace 'default' with 'delete'}}
struct FriendOfPrivatePrefix {
  // A friend has access to the prefix operator.
  friend PrivatePrefix operator--(PrivatePrefix &, int) = default; // #FriendOfPrivatePrefix
  // expected-warning@#FriendOfPrivatePrefix {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#FriendOfPrivatePrefix {{defaulted 'operator--' is implicitly deleted because there is no viable prefix 'operator--' for an lvalue of type 'PrivatePrefix'}}
  // expected-note@#FriendOfPrivatePrefix {{replace 'default' with 'delete'}}
};

struct DeletedCopy {
  DeletedCopy &operator++();
  DeletedCopy(const DeletedCopy &) = delete; // expected-note {{'DeletedCopy' has been explicitly marked deleted here}}
  DeletedCopy operator++(int) = default; // #DeletedCopy
  // expected-warning@#DeletedCopy {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#DeletedCopy {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted constructor}}
  // expected-note@#DeletedCopy {{replace 'default' with 'delete'}}
};

struct MoveOnly {
  MoveOnly &operator++();
  MoveOnly(MoveOnly &&); // expected-note {{copy constructor is implicitly deleted because 'MoveOnly' has a user-declared move constructor}}
  MoveOnly operator++(int) = default; // #MoveOnly
  // expected-warning@#MoveOnly {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#MoveOnly {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted constructor}}
  // expected-note@#MoveOnly {{replace 'default' with 'delete'}}
};

struct AmbiguousCopy {
  AmbiguousCopy &operator++();
  AmbiguousCopy(AmbiguousCopy &, int = 0);  // expected-note {{candidate constructor}}
  AmbiguousCopy(AmbiguousCopy &, long = 0); // expected-note {{candidate constructor}}
  AmbiguousCopy operator++(int) = default; // #AmbiguousCopy
  // expected-warning@#AmbiguousCopy {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#AmbiguousCopy {{defaulted 'operator++' is implicitly deleted because copy construction of an lvalue of type 'AmbiguousCopy' is ambiguous}}
  // expected-note@#AmbiguousCopy {{replace 'default' with 'delete'}}
};

class PrivateCopy {
  PrivateCopy(const PrivateCopy &); // expected-note {{declared private here}}

public:
  PrivateCopy &operator++();
  PrivateCopy &operator--();
  PrivateCopy operator++(int) = default; // OK, a member has access to the copy constructor
};
PrivateCopy operator--(PrivateCopy &, int) = default; // #PrivateCopy
// expected-warning@#PrivateCopy {{explicitly defaulted postfix decrement operator is implicitly deleted}}
// expected-note@#PrivateCopy {{defaulted 'operator--' is implicitly deleted because it would invoke a private constructor of 'PrivateCopy'}}
// expected-note@#PrivateCopy {{replace 'default' with 'delete'}}

struct DeletedDestructor {
  DeletedDestructor &operator++();
  ~DeletedDestructor() = delete; // expected-note {{'~DeletedDestructor' has been explicitly marked deleted here}}
  DeletedDestructor operator++(int) = default; // #DeletedDestructor
  // expected-warning@#DeletedDestructor {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#DeletedDestructor {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted destructor}}
  // expected-note@#DeletedDestructor {{replace 'default' with 'delete'}}
};

class PrivateDestructor {
  ~PrivateDestructor(); // expected-note {{declared private here}}

public:
  PrivateDestructor &operator++();
  PrivateDestructor(const PrivateDestructor &);
  PrivateDestructor operator++(int) = default; // OK, a member has access to the destructor
};
PrivateDestructor operator++(PrivateDestructor &, int) = default; // #PrivateDestructor
// expected-warning@#PrivateDestructor {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#PrivateDestructor {{defaulted 'operator++' is implicitly deleted because it would invoke a private destructor of 'PrivateDestructor'}}
// expected-note@#PrivateDestructor {{replace 'default' with 'delete'}}

// A function that is not defaulted on its first declaration cannot be deleted.
struct NotFirst {
  NotFirst &operator++();
  NotFirst(const NotFirst &) = delete; // expected-note {{'NotFirst' has been explicitly marked deleted here}}
  NotFirst operator++(int);
};
NotFirst NotFirst::operator++(int) = default; // expected-error {{defaulting this postfix increment operator would delete it after its first declaration}}
// expected-note@-1 {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted constructor}}

// No warning is produced for a template instantiation, but the function is
// still deleted.
template <typename T> struct Wrapper {
  T v;
  Wrapper operator++(int) = default; // #Wrapper
  // expected-note@#Wrapper {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#Wrapper {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'Wrapper<int>'}}
};
void use_wrapper(Wrapper<int> w) {
  w++; // expected-error {{object of type 'Wrapper<int>' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}

// -Wdefaulted-function-deleted can be disabled.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
struct Quiet {
  Quiet operator++(int) = default;
};
#pragma clang diagnostic pop
