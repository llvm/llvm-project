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
  // expected-error@#DeletedCopy {{call to deleted constructor of 'DeletedCopy'}}
  // expected-note@#DeletedCopy {{replace 'default' with 'delete'}}
};

struct MoveOnly {
  MoveOnly &operator++();
  MoveOnly(MoveOnly &&); // expected-note {{copy constructor is implicitly deleted because 'MoveOnly' has a user-declared move constructor}}
  MoveOnly operator++(int) = default; // #MoveOnly
  // expected-warning@#MoveOnly {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-error@#MoveOnly {{call to implicitly-deleted copy constructor of 'MoveOnly'}}
  // expected-note@#MoveOnly {{replace 'default' with 'delete'}}
};

struct AmbiguousCopy {
  AmbiguousCopy &operator++();
  AmbiguousCopy(AmbiguousCopy &, int = 0);  // expected-note {{candidate constructor}}
  AmbiguousCopy(AmbiguousCopy &, long = 0); // expected-note {{candidate constructor}}
  AmbiguousCopy operator++(int) = default; // #AmbiguousCopy
  // expected-warning@#AmbiguousCopy {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-error@#AmbiguousCopy {{call to constructor of 'AmbiguousCopy' is ambiguous}}
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
// expected-error@-1 {{call to deleted constructor of 'NotFirst'}}

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

// The same notes are produced for the postfix decrement operator.
struct DeletedPrefixDecrement {
  DeletedPrefixDecrement &operator--() = delete; // expected-note {{'operator--' has been explicitly marked deleted here}}
  DeletedPrefixDecrement operator--(int) = default; // #DeletedPrefixDecrement
  // expected-warning@#DeletedPrefixDecrement {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#DeletedPrefixDecrement {{defaulted 'operator--' is implicitly deleted because it would invoke a deleted prefix 'operator--'}}
  // expected-note@#DeletedPrefixDecrement {{replace 'default' with 'delete'}}
};
struct AmbiguousBuiltinDecrement {
  operator int &();
  operator long &();
  AmbiguousBuiltinDecrement operator--(int) = default; // #AmbiguousBuiltinDecrement
  // expected-warning@#AmbiguousBuiltinDecrement {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#AmbiguousBuiltinDecrement {{defaulted 'operator--' is implicitly deleted because prefix 'operator--' for an lvalue of type 'AmbiguousBuiltinDecrement' is ambiguous}}
  // expected-note@#AmbiguousBuiltinDecrement {{built-in candidate operator--(int &)}}
  // expected-note@#AmbiguousBuiltinDecrement {{built-in candidate operator--(long &)}}
  // expected-note@#AmbiguousBuiltinDecrement {{replace 'default' with 'delete'}}
};
class PrivatePrefixDecrement {
  PrivatePrefixDecrement &operator--(); // expected-note {{implicitly declared private here}}
};
PrivatePrefixDecrement operator--(PrivatePrefixDecrement &, int) = default; // #PrivatePrefixDecrementNonMember
// expected-warning@#PrivatePrefixDecrementNonMember {{explicitly defaulted postfix decrement operator is implicitly deleted}}
// expected-note@#PrivatePrefixDecrementNonMember {{defaulted 'operator--' is implicitly deleted because it would invoke a private prefix 'operator--' of 'PrivatePrefixDecrement'}}
// expected-note@#PrivatePrefixDecrementNonMember {{replace 'default' with 'delete'}}

// Protected members.
class ProtectedPrefix {
protected:
  ProtectedPrefix &operator++(); // expected-note {{declared protected here}}
  ProtectedPrefix &operator--(); // expected-note {{declared protected here}}
};
ProtectedPrefix operator++(ProtectedPrefix &, int) = default; // #ProtectedPrefixIncrement
// expected-warning@#ProtectedPrefixIncrement {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#ProtectedPrefixIncrement {{defaulted 'operator++' is implicitly deleted because it would invoke a protected prefix 'operator++' of 'ProtectedPrefix'}}
// expected-note@#ProtectedPrefixIncrement {{replace 'default' with 'delete'}}
ProtectedPrefix operator--(ProtectedPrefix &, int) = default; // #ProtectedPrefixDecrement
// expected-warning@#ProtectedPrefixDecrement {{explicitly defaulted postfix decrement operator is implicitly deleted}}
// expected-note@#ProtectedPrefixDecrement {{defaulted 'operator--' is implicitly deleted because it would invoke a protected prefix 'operator--' of 'ProtectedPrefix'}}
// expected-note@#ProtectedPrefixDecrement {{replace 'default' with 'delete'}}
// A friend of a derived class cannot use the protected member on an object of
// the base class type.
class ProtectedBase {
protected:
  ProtectedBase &operator--(); // expected-note {{can only access this member on an object of type 'DerivedFromProtectedBase'}}
};
struct DerivedFromProtectedBase : ProtectedBase {
  friend ProtectedBase operator--(ProtectedBase &, int) = default; // #DerivedFromProtectedBase
  // expected-warning@#DerivedFromProtectedBase {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#DerivedFromProtectedBase {{defaulted 'operator--' is implicitly deleted because it would invoke a protected prefix 'operator--' of 'ProtectedBase'}}
  // expected-note@#DerivedFromProtectedBase {{replace 'default' with 'delete'}}
};

class ProtectedCopy {
protected:
  ProtectedCopy(const ProtectedCopy &); // expected-note {{declared protected here}}

public:
  ProtectedCopy &operator++();
  ProtectedCopy operator++(int) = default; // OK, a member has access to the copy constructor
};
ProtectedCopy operator++(ProtectedCopy &, int) = default; // #ProtectedCopy
// expected-warning@#ProtectedCopy {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#ProtectedCopy {{defaulted 'operator++' is implicitly deleted because it would invoke a protected constructor of 'ProtectedCopy'}}
// expected-note@#ProtectedCopy {{replace 'default' with 'delete'}}

class ProtectedDestructor {
protected:
  ~ProtectedDestructor(); // expected-note {{declared protected here}}

public:
  ProtectedDestructor &operator--();
  ProtectedDestructor(const ProtectedDestructor &);
  ProtectedDestructor operator--(int) = default; // OK, a member has access to the destructor
};
ProtectedDestructor operator--(ProtectedDestructor &, int) = default; // #ProtectedDestructor
// expected-warning@#ProtectedDestructor {{explicitly defaulted postfix decrement operator is implicitly deleted}}
// expected-note@#ProtectedDestructor {{defaulted 'operator--' is implicitly deleted because it would invoke a protected destructor of 'ProtectedDestructor'}}
// expected-note@#ProtectedDestructor {{replace 'default' with 'delete'}}

struct NotFirstDecrement {
  NotFirstDecrement &operator--();
  ~NotFirstDecrement() = delete; // expected-note {{'~NotFirstDecrement' has been explicitly marked deleted here}}
  NotFirstDecrement operator--(int);
};
NotFirstDecrement NotFirstDecrement::operator--(int) = default; // expected-error {{defaulting this postfix decrement operator would delete it after its first declaration}}
// expected-note@-1 {{defaulted 'operator--' is implicitly deleted because it would invoke a deleted destructor}}

// Postfix operators with an explicit object parameter.
struct ExplicitObjectNoPrefix {
  ExplicitObjectNoPrefix operator++(this ExplicitObjectNoPrefix &, int) = default; // #ExplicitObjectNoPrefix
  // expected-warning@#ExplicitObjectNoPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#ExplicitObjectNoPrefix 2 {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'ExplicitObjectNoPrefix'}}
  // expected-note@#ExplicitObjectNoPrefix {{replace 'default' with 'delete'}}
  // expected-note@#ExplicitObjectNoPrefix {{explicitly defaulted function was implicitly deleted here}}
  ExplicitObjectNoPrefix operator--(this ExplicitObjectNoPrefix &&, int) = default; // #ExplicitObjectNoPrefixDecrement
  // expected-warning@#ExplicitObjectNoPrefixDecrement {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#ExplicitObjectNoPrefixDecrement 2 {{defaulted 'operator--' is implicitly deleted because there is no viable prefix 'operator--' for an lvalue of type 'ExplicitObjectNoPrefix'}}
  // expected-note@#ExplicitObjectNoPrefixDecrement {{replace 'default' with 'delete'}}
  // expected-note@#ExplicitObjectNoPrefixDecrement {{explicitly defaulted function was implicitly deleted here}}
};
void use_explicit_object_no_prefix(ExplicitObjectNoPrefix e) {
  e++; // expected-error {{object of type 'ExplicitObjectNoPrefix' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
  static_cast<ExplicitObjectNoPrefix &&>(e)--; // expected-error {{object of type 'ExplicitObjectNoPrefix' cannot be decremented because its defaulted postfix decrement operator is implicitly deleted}}
}
// An explicit object member function is a member, so it has access to the
// private members of its class.
class ExplicitObjectPrivate {
  ExplicitObjectPrivate(const ExplicitObjectPrivate &);
  ~ExplicitObjectPrivate();
  ExplicitObjectPrivate &operator++();

public:
  ExplicitObjectPrivate operator++(this ExplicitObjectPrivate &, int) = default; // OK
};
struct ExplicitObjectDeletedPrefix {
  ExplicitObjectDeletedPrefix &operator++() = delete; // expected-note {{'operator++' has been explicitly marked deleted here}}
  ExplicitObjectDeletedPrefix operator++(this ExplicitObjectDeletedPrefix &, int) = default; // #ExplicitObjectDeletedPrefix
  // expected-warning@#ExplicitObjectDeletedPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#ExplicitObjectDeletedPrefix {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted prefix 'operator++'}}
  // expected-note@#ExplicitObjectDeletedPrefix {{replace 'default' with 'delete'}}
};

// Prefix operators with an explicit object parameter. The prefix operator is
// applied to an lvalue of type C, so an explicit object parameter of type
// "rvalue reference to C" is not viable.
struct ExplicitObjectPrefix {
  ExplicitObjectPrefix &operator++(this ExplicitObjectPrefix &);
  ExplicitObjectPrefix operator++(int) = default; // OK
};
struct RvalueExplicitObjectPrefix {
  RvalueExplicitObjectPrefix &operator++(this RvalueExplicitObjectPrefix &&);
  RvalueExplicitObjectPrefix operator++(int) = default; // #RvalueExplicitObjectPrefix
  // expected-warning@#RvalueExplicitObjectPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#RvalueExplicitObjectPrefix {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'RvalueExplicitObjectPrefix'}}
  // expected-note@#RvalueExplicitObjectPrefix {{replace 'default' with 'delete'}}
};
struct DeletedExplicitObjectPrefix {
  DeletedExplicitObjectPrefix &operator--(this DeletedExplicitObjectPrefix &) = delete; // expected-note {{'operator--' has been explicitly marked deleted here}}
  DeletedExplicitObjectPrefix operator--(int) = default; // #DeletedExplicitObjectPrefix
  // expected-warning@#DeletedExplicitObjectPrefix {{explicitly defaulted postfix decrement operator is implicitly deleted}}
  // expected-note@#DeletedExplicitObjectPrefix {{defaulted 'operator--' is implicitly deleted because it would invoke a deleted prefix 'operator--'}}
  // expected-note@#DeletedExplicitObjectPrefix {{replace 'default' with 'delete'}}
};
class PrivateExplicitObjectPrefix {
  PrivateExplicitObjectPrefix &operator++(this PrivateExplicitObjectPrefix &); // expected-note {{implicitly declared private here}}
};
PrivateExplicitObjectPrefix operator++(PrivateExplicitObjectPrefix &, int) = default; // #PrivateExplicitObjectPrefix
// expected-warning@#PrivateExplicitObjectPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#PrivateExplicitObjectPrefix {{defaulted 'operator++' is implicitly deleted because it would invoke a private prefix 'operator++' of 'PrivateExplicitObjectPrefix'}}
// expected-note@#PrivateExplicitObjectPrefix {{replace 'default' with 'delete'}}
// A deduced explicit object parameter makes the prefix operator a template.
// The deleted specialization is selected by overload resolution.
struct DeletedDeducedPrefix {
  template <typename Self> DeletedDeducedPrefix &operator++(this Self &&) = delete; // expected-note {{'operator++<DeletedDeducedPrefix &>' has been explicitly marked deleted here}}
  DeletedDeducedPrefix operator++(int) = default; // #DeletedDeducedPrefix
  // expected-warning@#DeletedDeducedPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#DeletedDeducedPrefix {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted prefix 'operator++'}}
  // expected-note@#DeletedDeducedPrefix {{replace 'default' with 'delete'}}
};

// Constrained prefix operators. An unsatisfied constraint makes the candidate
// non-viable.
template <typename T> concept Small = sizeof(T) <= 4;
template <typename T> struct ConstrainedPrefix {
  T v;
  ConstrainedPrefix &operator++() requires Small<T>;
  ConstrainedPrefix &operator--(this ConstrainedPrefix &) requires Small<T>;
  ConstrainedPrefix operator++(int) = default; // #ConstrainedPrefixIncrement
  // expected-note@#ConstrainedPrefixIncrement {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#ConstrainedPrefixIncrement {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'ConstrainedPrefix<long long>'}}
  friend ConstrainedPrefix operator--(ConstrainedPrefix &, int) = default; // #ConstrainedPrefixDecrement
  // expected-note@#ConstrainedPrefixDecrement {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#ConstrainedPrefixDecrement {{defaulted 'operator--' is implicitly deleted because there is no viable prefix 'operator--' for an lvalue of type 'ConstrainedPrefix<long long>'}}
};
void use_constrained_prefix(ConstrainedPrefix<int> a, ConstrainedPrefix<long long> b) {
  a++; // OK
  a--; // OK
  b++; // expected-error {{object of type 'ConstrainedPrefix<long long>' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
  b--; // expected-error {{object of type 'ConstrainedPrefix<long long>' cannot be decremented because its defaulted postfix decrement operator is implicitly deleted}}
}
struct ConstrainedNonMemberPrefix { long long a, b; };
template <typename T> requires (__is_same(T, ConstrainedNonMemberPrefix) && Small<T>) T &operator++(T &);
ConstrainedNonMemberPrefix operator++(ConstrainedNonMemberPrefix &, int) = default; // #ConstrainedNonMemberPrefix
// expected-warning@#ConstrainedNonMemberPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@#ConstrainedNonMemberPrefix {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'ConstrainedNonMemberPrefix'}}
// expected-note@#ConstrainedNonMemberPrefix {{replace 'default' with 'delete'}}
// A deleted candidate is still viable if its constraints are satisfied.
struct DeletedConstrainedPrefix {};
template <typename T> requires __is_same(T, DeletedConstrainedPrefix) T &operator--(T &) = delete; // expected-note {{'operator--<DeletedConstrainedPrefix>' has been explicitly marked deleted here}}
DeletedConstrainedPrefix operator--(DeletedConstrainedPrefix &, int) = default; // #DeletedConstrainedPrefix
// expected-warning@#DeletedConstrainedPrefix {{explicitly defaulted postfix decrement operator is implicitly deleted}}
// expected-note@#DeletedConstrainedPrefix {{defaulted 'operator--' is implicitly deleted because it would invoke a deleted prefix 'operator--'}}
// expected-note@#DeletedConstrainedPrefix {{replace 'default' with 'delete'}}
// The more constrained prefix operator is selected.
template <typename T> struct MoreConstrainedPrefix {
  MoreConstrainedPrefix &operator++() requires Small<T>;
  MoreConstrainedPrefix &operator++() requires Small<T> && __is_same(T, int) = delete; // expected-note {{'operator++' has been explicitly marked deleted here}}
  MoreConstrainedPrefix operator++(int) = default; // #MoreConstrainedPrefix
  // expected-note@#MoreConstrainedPrefix {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#MoreConstrainedPrefix {{defaulted 'operator++' is implicitly deleted because it would invoke a deleted prefix 'operator++'}}
};
void use_more_constrained_prefix(MoreConstrainedPrefix<short> a, MoreConstrainedPrefix<int> b) {
  a++; // OK
  b++; // expected-error {{object of type 'MoreConstrainedPrefix<int>' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}

// Constrained postfix operators. The constraints of the postfix operator do
// not affect whether it is defined as deleted, only whether it is viable.
template <typename T> struct ConstrainedPostfix {
  ConstrainedPostfix operator++(int) requires Small<T> = default; // #ConstrainedPostfix
  // expected-note@#ConstrainedPostfix {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#ConstrainedPostfix {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'ConstrainedPostfix<int>'}}
};
void use_constrained_postfix(ConstrainedPostfix<int> a, ConstrainedPostfix<long long> b) {
  a++; // expected-error {{object of type 'ConstrainedPostfix<int>' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
  b++; // expected-error {{cannot increment value of type 'ConstrainedPostfix<long long>'}}
}

// Constrained special member functions. The copy constructor and destructor
// are selected by overload resolution, so their constraints decide whether the
// defaulted postfix operator is deleted.
template <typename T> struct ConstrainedCopy {
  ConstrainedCopy() = default;
  ConstrainedCopy(const ConstrainedCopy &) requires Small<T> = default;
  ConstrainedCopy(const ConstrainedCopy &) = delete; // expected-note {{'ConstrainedCopy' has been explicitly marked deleted here}}
  ConstrainedCopy &operator++();
  ConstrainedCopy operator++(int) = default; // #ConstrainedCopy
  // expected-note@#ConstrainedCopy {{explicitly defaulted function was implicitly deleted here}}
  // expected-error@#ConstrainedCopy {{call to deleted constructor of 'ConstrainedCopy<long long>'}}
};
void use_constrained_copy(ConstrainedCopy<int> &a, ConstrainedCopy<long long> &b) {
  a++; // OK
  b++; // expected-error {{object of type 'ConstrainedCopy<long long>' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}
template <typename T> struct ConstrainedDestructor {
  ~ConstrainedDestructor() requires Small<T> = default;
  ~ConstrainedDestructor() = delete; // expected-note {{'~ConstrainedDestructor' has been explicitly marked deleted here}}
  ConstrainedDestructor &operator--();
  friend ConstrainedDestructor operator--(ConstrainedDestructor &, int) = default; // #ConstrainedDestructor
  // expected-note@#ConstrainedDestructor {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#ConstrainedDestructor {{defaulted 'operator--' is implicitly deleted because it would invoke a deleted destructor}}
};
void use_constrained_destructor(ConstrainedDestructor<int> &a, ConstrainedDestructor<long long> &b) {
  a--; // OK
  b--; // expected-error {{object of type 'ConstrainedDestructor<long long>' cannot be decremented because its defaulted postfix decrement operator is implicitly deleted}}
}
