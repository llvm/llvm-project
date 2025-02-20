// RUN: %clang_cc1 -std=c++2c -verify %s

class Trivial {};
struct NonRelocatable {
    ~NonRelocatable();
};
static NonRelocatable NonRelocatable_g;

class A trivially_relocatable_if_eligible {};
class B trivially_relocatable_if_eligible : Trivial{};
class C trivially_relocatable_if_eligible {
    int a;
    void* b;
    int c[3];
    Trivial d[3];
    NonRelocatable& e = NonRelocatable_g;
};
class D trivially_relocatable_if_eligible : Trivial {};
class E trivially_relocatable_if_eligible : virtual Trivial {};

class F trivially_relocatable_if_eligible : NonRelocatable {};

class I trivially_relocatable_if_eligible {
    NonRelocatable a;
    NonRelocatable b[1];
    const NonRelocatable c;
    const NonRelocatable d[1];
};

class J trivially_relocatable_if_eligible:  virtual Trivial, NonRelocatable {
    NonRelocatable a;
};

class G trivially_relocatable_if_eligible {
    G(G&&);
};

class H trivially_relocatable_if_eligible {
    ~H();
};

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}
static_assert(__builtin_is_cpp_trivially_relocatable(Incomplete));  // expected-error {{incomplete type 'Incomplete' used in type trait expression}}
static_assert(__builtin_is_cpp_trivially_relocatable(Trivial));
static_assert(__builtin_is_cpp_trivially_relocatable(G));
static_assert(__builtin_is_cpp_trivially_relocatable(H));
static_assert(__builtin_is_cpp_trivially_relocatable(int));
static_assert(__builtin_is_cpp_trivially_relocatable(void*));
static_assert(!__builtin_is_cpp_trivially_relocatable(int&));
static_assert(!__builtin_is_cpp_trivially_relocatable(Trivial&));
static_assert(__builtin_is_cpp_trivially_relocatable(const Trivial));
static_assert(__builtin_is_cpp_trivially_relocatable(Trivial[1]));

struct UserDtr {
    ~UserDtr();
};

struct DefaultedDtr {
    ~DefaultedDtr() = default;
};
struct UserMoveWithDefaultCopy {
    UserMoveWithDefaultCopy(UserMoveWithDefaultCopy&&);
    UserMoveWithDefaultCopy(const UserMoveWithDefaultCopy&) = default;
};

struct UserMove{
    UserMove(UserMove&&);
};

struct UserMoveDefault{
    UserMoveDefault(UserMoveDefault&&) = default;
};

struct UserCopy{
    UserCopy(const UserCopy&);
};

struct UserCopyDefault{
    UserCopyDefault(const UserCopyDefault&) = default;
};


struct UserDeletedMove{
    UserDeletedMove(UserDeletedMove&) = delete;
    UserDeletedMove(const UserDeletedMove&) = default;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(UserDtr));
static_assert(__builtin_is_cpp_trivially_relocatable(DefaultedDtr));
static_assert(!__builtin_is_cpp_trivially_relocatable(UserMoveWithDefaultCopy));
static_assert(!__builtin_is_cpp_trivially_relocatable(UserMove));
static_assert(!__builtin_is_cpp_trivially_relocatable(UserCopy));
static_assert(__builtin_is_cpp_trivially_relocatable(UserMoveDefault));
static_assert(__builtin_is_cpp_trivially_relocatable(UserCopyDefault));
static_assert(__builtin_is_cpp_trivially_relocatable(UserDeletedMove));

template <typename T>
class TestDependentErrors trivially_relocatable_if_eligible : T {};
TestDependentErrors<Trivial> Ok;
TestDependentErrors<NonRelocatable> Err;

struct DeletedMove {
    DeletedMove(DeletedMove&&) = delete;
};
struct DeletedCopy {
    DeletedCopy(const DeletedCopy&) = delete;
};
struct DeletedMoveAssign {
    DeletedMoveAssign& operator=(DeletedMoveAssign&&) = delete;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(DeletedMove));
static_assert(!__builtin_is_cpp_trivially_relocatable(DeletedCopy));
static_assert(!__builtin_is_cpp_trivially_relocatable(DeletedMoveAssign));

union U {
    G g;
};
static_assert(!__is_trivially_copyable(U));
static_assert(__builtin_is_cpp_trivially_relocatable(U));


namespace replaceable {

struct DeletedMove {
    DeletedMove(DeletedMove&&) = delete;
};
struct DeletedCopy {
    DeletedCopy(const DeletedCopy&) = delete;
};
struct DeletedMoveAssign {
    DeletedMoveAssign& operator=(DeletedMoveAssign&&) = delete;
};

struct DefaultedMove {
    DefaultedMove(DefaultedMove&&) = default;
    DefaultedMove& operator=(DefaultedMove&&) = default;
};
struct DefaultedCopy {
    DefaultedCopy(const DefaultedCopy&) = default;
    DefaultedCopy(DefaultedCopy&&) = default;
    DefaultedCopy& operator=(DefaultedCopy&&) = default;
};
struct DefaultedMoveAssign {
    DefaultedMoveAssign(DefaultedMoveAssign&&) = default;
    DefaultedMoveAssign& operator=(DefaultedMoveAssign&&) = default;
};

struct UserProvidedMove {
    UserProvidedMove(UserProvidedMove&&){};
};
struct UserProvidedCopy {
    UserProvidedCopy(const UserProvidedCopy&) {};
};
struct UserProvidedMoveAssign {
    UserProvidedMoveAssign& operator=(const UserProvidedMoveAssign&){return *this;};
};

struct Empty{};
static_assert(__builtin_is_replaceable(Empty));
struct S1 replaceable_if_eligible{};
static_assert(__builtin_is_replaceable(S1));

static_assert(__builtin_is_replaceable(DefaultedMove));
static_assert(__builtin_is_replaceable(DefaultedCopy));
static_assert(__builtin_is_replaceable(DefaultedMoveAssign));

static_assert(!__builtin_is_replaceable(DeletedMove));
static_assert(!__builtin_is_replaceable(DeletedCopy));
static_assert(!__builtin_is_replaceable(DeletedMoveAssign));

static_assert(!__builtin_is_replaceable(UserProvidedMove));
static_assert(!__builtin_is_replaceable(UserProvidedCopy));
static_assert(!__builtin_is_replaceable(UserProvidedMoveAssign));

using NotReplaceable = DeletedMove;

template <typename T>
struct S {
    T t;
};

template <typename T>
struct WithBase : T{};

template <typename T>
struct WithVBase : virtual T{};

struct WithVirtual {
    virtual ~WithVirtual() = default;
    WithVirtual(WithVirtual&&) = default;
    WithVirtual& operator=(WithVirtual&&) = default;
};

static_assert(__builtin_is_replaceable(S<int>));
static_assert(__builtin_is_replaceable(S<volatile int>));
static_assert(!__builtin_is_replaceable(S<const int>));
static_assert(!__builtin_is_replaceable(S<const int&>));
static_assert(!__builtin_is_replaceable(S<int&>));
static_assert(__builtin_is_replaceable(S<int[2]>));
static_assert(!__builtin_is_replaceable(S<const int[2]>));
static_assert(__builtin_is_replaceable(WithBase<S<int>>));
static_assert(!__builtin_is_replaceable(WithBase<S<const int>>));
static_assert(!__builtin_is_replaceable(WithBase<UserProvidedMove>));
static_assert(!__builtin_is_replaceable(WithVBase<S<int>>));
static_assert(!__builtin_is_replaceable(WithVBase<S<const int>>));
static_assert(!__builtin_is_replaceable(WithVBase<UserProvidedMove>));
static_assert(__builtin_is_replaceable(WithVirtual));


struct U1 replaceable_if_eligible {
    ~U1() = delete;
    U1(U1&&) = default;
    U1& operator=(U1&&) = default;

};
static_assert(__builtin_is_replaceable(U1));

struct U2 replaceable_if_eligible {
    U2(const U2&) = delete;
};
static_assert(!__builtin_is_replaceable(U2));


template <typename T>
struct WithVBaseExplicit replaceable_if_eligible : virtual T{};
static_assert(__builtin_is_replaceable(WithVBaseExplicit<S<int>>)); // expected-error {{failed}}

struct S42 trivially_relocatable_if_eligible replaceable_if_eligible {
    S42(S42&&);
    S42& operator=(S42&&) = default;
};
struct S43 trivially_relocatable_if_eligible replaceable_if_eligible {
    S43(S43&&) = default;
    S43& operator=(S43&&);
};


struct Copyable1Explicit replaceable_if_eligible {
   Copyable1Explicit(Copyable1Explicit const &) = default;
};

struct Copyable1 {
   Copyable1(Copyable1 const &) = default;
};


struct CopyAssign1Explicit replaceable_if_eligible {
   CopyAssign1Explicit & operator=(const CopyAssign1Explicit&) = default;
};

struct CopyAssign1 {
   CopyAssign1 & operator=(CopyAssign1 const &) = default;
};

}


void test__builtin_trivially_relocate() {
    struct S{ ~S();};
    struct R {};
    __builtin_trivially_relocate(); //expected-error {{too few arguments to function call, expected 3, have 0}}
    __builtin_trivially_relocate(0, 0, 0, 0); //expected-error {{too many arguments to function call, expected 3, have 4}}
    __builtin_trivially_relocate(0, 0, 0); //expected-error {{argument to '__builtin_trivially_relocate' must be a pointer}}
    __builtin_trivially_relocate((const int*)0, 0, 0); //expected-error {{argument to '__builtin_trivially_relocate' must be non-const}}
    __builtin_trivially_relocate((S*)0, 0, 0); //expected-error {{argument to '__builtin_trivially_relocate' must be relocatable}}
    __builtin_trivially_relocate((int*)0, 0, 0); //expected-error {{first and second arguments to '__builtin_trivially_relocate' must be of the same type}}

    __builtin_trivially_relocate((int*)0, (int*)0, (int*)0); // expected-error{{cannot initialize a value of type 'unsigned long' with an rvalue of type 'int *'}}
    __builtin_trivially_relocate((int*)0, (int*)0, 0);
    __builtin_trivially_relocate((R*)0, (R*)0, 0);
}
