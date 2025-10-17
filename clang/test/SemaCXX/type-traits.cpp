// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++11 -fblocks -Wno-deprecated-builtins -Wno-defaulted-function-deleted -Wno-c++17-extensions  %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++14 -fblocks -Wno-deprecated-builtins -Wno-defaulted-function-deleted -Wno-c++17-extensions  %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++17 -fblocks -Wno-deprecated-builtins -Wno-defaulted-function-deleted  %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++20 -fblocks -Wno-deprecated-builtins -Wno-defaulted-function-deleted  %s


struct NonPOD { NonPOD(int); };
typedef NonPOD NonPODAr[10];
typedef NonPOD NonPODArNB[];
typedef NonPOD NonPODArMB[10][2];

// PODs
enum Enum { EV };
enum SignedEnum : signed int { };
enum UnsignedEnum : unsigned int { };
enum class EnumClass { EV };
enum class SignedEnumClass : signed int {};
enum class UnsignedEnumClass : unsigned int {};
struct POD { Enum e; int i; float f; NonPOD* p; };
struct Empty {};
struct IncompleteStruct; // expected-note {{forward declaration of 'IncompleteStruct'}}
typedef Empty EmptyAr[10];
typedef Empty EmptyArNB[];
typedef Empty EmptyArMB[1][2];
typedef int Int;
typedef Int IntAr[10];
typedef Int IntArNB[];
typedef Int IntArZero[0];
class Statics { static int priv; static NonPOD np; };
union EmptyUnion {};
union IncompleteUnion; // expected-note {{forward declaration of 'IncompleteUnion'}}
union Union { int i; float f; };
struct HasFunc { void f (); };
struct HasOp { void operator *(); };
struct HasConv { operator int(); };
struct HasAssign { void operator =(int); };

struct HasAnonymousUnion {
  union {
    int i;
    float f;
  };
};

typedef int Vector __attribute__((vector_size(16)));
typedef int VectorExt __attribute__((ext_vector_type(4)));

using ComplexFloat = _Complex float;
using ComplexInt = _Complex int;

// Not PODs
typedef const void cvoid;
struct Derives : POD {};
typedef Derives DerivesAr[10];
typedef Derives DerivesArNB[];
struct DerivesEmpty : Empty {};
struct HasCons { HasCons(int); };
struct HasDefaultCons { HasDefaultCons() = default; };
struct HasExplicitDefaultCons { explicit HasExplicitDefaultCons() = default; };
struct HasInheritedCons : HasDefaultCons { using HasDefaultCons::HasDefaultCons; };
struct HasNoInheritedCons : HasCons {};
struct HasCopyAssign { HasCopyAssign operator =(const HasCopyAssign&); };
struct HasMoveAssign { HasMoveAssign operator =(const HasMoveAssign&&); };
struct HasNoThrowMoveAssign {
  HasNoThrowMoveAssign& operator=(
    const HasNoThrowMoveAssign&&) throw(); };
struct HasNoExceptNoThrowMoveAssign {
  HasNoExceptNoThrowMoveAssign& operator=(
    const HasNoExceptNoThrowMoveAssign&&) noexcept;
};
struct HasThrowMoveAssign {
  HasThrowMoveAssign& operator=(const HasThrowMoveAssign&&)
#if __cplusplus <= 201402L
  throw(POD);
#else
  noexcept(false);
#endif
};


struct HasNoExceptFalseMoveAssign {
  HasNoExceptFalseMoveAssign& operator=(
    const HasNoExceptFalseMoveAssign&&) noexcept(false); };
struct HasMoveCtor { HasMoveCtor(const HasMoveCtor&&); };
struct HasMemberMoveCtor { HasMoveCtor member; };
struct HasMemberMoveAssign { HasMoveAssign member; };
struct HasStaticMemberMoveCtor { static HasMoveCtor member; };
struct HasStaticMemberMoveAssign { static HasMoveAssign member; };
struct HasMemberThrowMoveAssign { HasThrowMoveAssign member; };
struct HasMemberNoExceptFalseMoveAssign {
  HasNoExceptFalseMoveAssign member; };
struct HasMemberNoThrowMoveAssign { HasNoThrowMoveAssign member; };
struct HasMemberNoExceptNoThrowMoveAssign {
  HasNoExceptNoThrowMoveAssign member; };

struct HasDefaultTrivialCopyAssign {
  HasDefaultTrivialCopyAssign &operator=(
    const HasDefaultTrivialCopyAssign&) = default;
};
struct TrivialMoveButNotCopy {
  TrivialMoveButNotCopy &operator=(TrivialMoveButNotCopy&&) = default;
  TrivialMoveButNotCopy &operator=(const TrivialMoveButNotCopy&);
};
struct NonTrivialDefault {
  NonTrivialDefault();
};

struct HasDest { ~HasDest(); };
class  HasPriv { int priv; };
class  HasProt { protected: int prot; };
struct HasRef { int i; int& ref; HasRef() : i(0), ref(i) {} };
struct HasRefAggregate { int i; int& ref; };
struct HasNonPOD { NonPOD np; };
struct HasVirt { virtual void Virt() {}; };
typedef NonPOD NonPODAr[10];
typedef HasVirt VirtAr[10];
typedef NonPOD NonPODArNB[];
union NonPODUnion { int i; Derives n; };
struct DerivesHasCons : HasCons {};
struct DerivesHasCopyAssign : HasCopyAssign {};
struct DerivesHasMoveAssign : HasMoveAssign {};
struct DerivesHasDest : HasDest {};
struct DerivesHasPriv : HasPriv {};
struct DerivesHasProt : HasProt {};
struct DerivesHasRef : HasRef {};
struct DerivesHasVirt : HasVirt {};
struct DerivesHasMoveCtor : HasMoveCtor {};

struct HasNoThrowCopyAssign {
  void operator =(const HasNoThrowCopyAssign&) throw();
};
struct HasMultipleCopyAssign {
  void operator =(const HasMultipleCopyAssign&) throw();
  void operator =(volatile HasMultipleCopyAssign&);
};
struct HasMultipleNoThrowCopyAssign {
  void operator =(const HasMultipleNoThrowCopyAssign&) throw();
  void operator =(volatile HasMultipleNoThrowCopyAssign&) throw();
};

struct HasNoThrowConstructor { HasNoThrowConstructor() throw(); };
struct HasNoThrowConstructorWithArgs {
  HasNoThrowConstructorWithArgs(HasCons i = HasCons(0)) throw();
};
struct HasMultipleDefaultConstructor1 {
  HasMultipleDefaultConstructor1() throw();
  HasMultipleDefaultConstructor1(int i = 0);
};
struct HasMultipleDefaultConstructor2 {
  HasMultipleDefaultConstructor2(int i = 0);
  HasMultipleDefaultConstructor2() throw();
};

struct HasNoThrowCopy { HasNoThrowCopy(const HasNoThrowCopy&) throw(); };
struct HasMultipleCopy {
  HasMultipleCopy(const HasMultipleCopy&) throw();
  HasMultipleCopy(volatile HasMultipleCopy&);
};
struct HasMultipleNoThrowCopy {
  HasMultipleNoThrowCopy(const HasMultipleNoThrowCopy&) throw();
  HasMultipleNoThrowCopy(volatile HasMultipleNoThrowCopy&) throw();
};

struct HasVirtDest { virtual ~HasVirtDest(); };
struct DerivedVirtDest : HasVirtDest {};
typedef HasVirtDest VirtDestAr[1];

class AllPrivate {
  AllPrivate() throw();
  AllPrivate(const AllPrivate&) throw();
  AllPrivate &operator=(const AllPrivate &) throw();
  ~AllPrivate() throw();
};

struct ThreeArgCtor {
  ThreeArgCtor(int*, char*, int);
};

struct VariadicCtor {
  template<typename...T> VariadicCtor(T...);
};

struct ThrowingDtor {
  ~ThrowingDtor()
#if __cplusplus <= 201402L
  throw(int);
#else
  noexcept(false);
#endif
};

struct NoExceptDtor {
  ~NoExceptDtor() noexcept(true);
};

struct NoThrowDtor {
  ~NoThrowDtor() throw();
};

struct ACompleteType {};
struct AnIncompleteType; // expected-note 1+ {{forward declaration of 'AnIncompleteType'}}
typedef AnIncompleteType AnIncompleteTypeAr[42];
typedef AnIncompleteType AnIncompleteTypeArNB[];
typedef AnIncompleteType AnIncompleteTypeArMB[1][10];

struct HasInClassInit {
  int x = 42;
};

struct HasPrivateBase : private ACompleteType {};
struct HasProtectedBase : protected ACompleteType {};
struct HasVirtBase : virtual ACompleteType {};

void is_pod()
{
  static_assert(__is_pod(int));
  static_assert(__is_pod(Enum));
  static_assert(__is_pod(POD));
  static_assert(__is_pod(Int));
  static_assert(__is_pod(IntAr));
  static_assert(__is_pod(Statics));
  static_assert(__is_pod(Empty));
  static_assert(__is_pod(EmptyUnion));
  static_assert(__is_pod(Union));
  static_assert(__is_pod(HasFunc));
  static_assert(__is_pod(HasOp));
  static_assert(__is_pod(HasConv));
  static_assert(__is_pod(HasAssign));
  static_assert(__is_pod(IntArNB));
  static_assert(__is_pod(HasAnonymousUnion));
  static_assert(__is_pod(Vector));
  static_assert(__is_pod(VectorExt));
  static_assert(__is_pod(Derives));
  static_assert(__is_pod(DerivesAr));
  static_assert(__is_pod(DerivesArNB));
  static_assert(__is_pod(DerivesEmpty));
  static_assert(__is_pod(HasPriv));
  static_assert(__is_pod(HasProt));
  static_assert(__is_pod(DerivesHasPriv));
  static_assert(__is_pod(DerivesHasProt));

  static_assert(!__is_pod(HasCons));
  static_assert(!__is_pod(HasCopyAssign));
  static_assert(!__is_pod(HasMoveAssign));
  static_assert(!__is_pod(HasDest));
  static_assert(!__is_pod(HasRef));
  static_assert(!__is_pod(HasVirt));
  static_assert(!__is_pod(DerivesHasCons));
  static_assert(!__is_pod(DerivesHasCopyAssign));
  static_assert(!__is_pod(DerivesHasMoveAssign));
  static_assert(!__is_pod(DerivesHasDest));
  static_assert(!__is_pod(DerivesHasRef));
  static_assert(!__is_pod(DerivesHasVirt));
  static_assert(!__is_pod(NonPOD));
  static_assert(!__is_pod(HasNonPOD));
  static_assert(!__is_pod(NonPODAr));
  static_assert(!__is_pod(NonPODArNB));
  static_assert(!__is_pod(void));
  static_assert(!__is_pod(cvoid));
// static_assert(!__is_pod(NonPODUnion));

  static_assert(__is_pod(ACompleteType));
  static_assert(!__is_pod(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_pod(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__is_pod(AnIncompleteType[1])); // expected-error {{incomplete type}}
}

typedef Empty EmptyAr[10];
struct Bit0 { int : 0; };
struct Bit0Cons { int : 0; Bit0Cons(); };
struct AnonBitOnly { int : 3; };
struct BitOnly { int x : 3; };
struct DerivesVirt : virtual POD {};

void is_empty()
{
  static_assert(__is_empty(Empty));
  static_assert(__is_empty(DerivesEmpty));
  static_assert(__is_empty(HasCons));
  static_assert(__is_empty(HasCopyAssign));
  static_assert(__is_empty(HasMoveAssign));
  static_assert(__is_empty(HasDest));
  static_assert(__is_empty(HasFunc));
  static_assert(__is_empty(HasOp));
  static_assert(__is_empty(HasConv));
  static_assert(__is_empty(HasAssign));
  static_assert(__is_empty(Bit0));
  static_assert(__is_empty(Bit0Cons));

  static_assert(!__is_empty(Int));
  static_assert(!__is_empty(POD));
  static_assert(!__is_empty(EmptyUnion));
  static_assert(!__is_empty(IncompleteUnion));
  static_assert(!__is_empty(EmptyAr));
  static_assert(!__is_empty(HasRef));
  static_assert(!__is_empty(HasVirt));
  static_assert(!__is_empty(AnonBitOnly));
  static_assert(!__is_empty(BitOnly));
  static_assert(!__is_empty(void));
  static_assert(!__is_empty(IntArNB));
  static_assert(!__is_empty(HasAnonymousUnion));
//  static_assert(!__is_empty(DerivesVirt));

  static_assert(__is_empty(ACompleteType));
  static_assert(!__is_empty(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_empty(AnIncompleteType[]));
  static_assert(!__is_empty(AnIncompleteType[1]));
}

typedef Derives ClassType;

void is_class()
{
  static_assert(__is_class(Derives));
  static_assert(__is_class(HasPriv));
  static_assert(__is_class(ClassType));
  static_assert(__is_class(HasAnonymousUnion));

  static_assert(!__is_class(int));
  static_assert(!__is_class(Enum));
  static_assert(!__is_class(Int));
  static_assert(!__is_class(IntAr));
  static_assert(!__is_class(DerivesAr));
  static_assert(!__is_class(Union));
  static_assert(!__is_class(cvoid));
  static_assert(!__is_class(IntArNB));
}

typedef Union UnionAr[10];
typedef Union UnionType;

void is_union()
{
  static_assert(__is_union(Union));
  static_assert(__is_union(UnionType));

  static_assert(!__is_union(int));
  static_assert(!__is_union(Enum));
  static_assert(!__is_union(Int));
  static_assert(!__is_union(IntAr));
  static_assert(!__is_union(UnionAr));
  static_assert(!__is_union(cvoid));
  static_assert(!__is_union(IntArNB));
  static_assert(!__is_union(HasAnonymousUnion));
}

typedef Enum EnumType;
typedef EnumClass EnumClassType;

void is_enum()
{
  static_assert(__is_enum(Enum));
  static_assert(__is_enum(EnumType));
  static_assert(__is_enum(SignedEnum));
  static_assert(__is_enum(UnsignedEnum));

  static_assert(__is_enum(EnumClass));
  static_assert(__is_enum(EnumClassType));
  static_assert(__is_enum(SignedEnumClass));
  static_assert(__is_enum(UnsignedEnumClass));

  static_assert(!__is_enum(int));
  static_assert(!__is_enum(Union));
  static_assert(!__is_enum(Int));
  static_assert(!__is_enum(IntAr));
  static_assert(!__is_enum(UnionAr));
  static_assert(!__is_enum(Derives));
  static_assert(!__is_enum(ClassType));
  static_assert(!__is_enum(cvoid));
  static_assert(!__is_enum(IntArNB));
  static_assert(!__is_enum(HasAnonymousUnion));
  static_assert(!__is_enum(AnIncompleteType));
  static_assert(!__is_enum(AnIncompleteTypeAr));
  static_assert(!__is_enum(AnIncompleteTypeArMB));
  static_assert(!__is_enum(AnIncompleteTypeArNB));
}

void is_scoped_enum() {
  static_assert(!__is_scoped_enum(Enum));
  static_assert(!__is_scoped_enum(EnumType));
  static_assert(!__is_scoped_enum(SignedEnum));
  static_assert(!__is_scoped_enum(UnsignedEnum));

  static_assert(__is_scoped_enum(EnumClass));
  static_assert(__is_scoped_enum(EnumClassType));
  static_assert(__is_scoped_enum(SignedEnumClass));
  static_assert(__is_scoped_enum(UnsignedEnumClass));

  static_assert(!__is_scoped_enum(int));
  static_assert(!__is_scoped_enum(Union));
  static_assert(!__is_scoped_enum(Int));
  static_assert(!__is_scoped_enum(IntAr));
  static_assert(!__is_scoped_enum(UnionAr));
  static_assert(!__is_scoped_enum(Derives));
  static_assert(!__is_scoped_enum(ClassType));
  static_assert(!__is_scoped_enum(cvoid));
  static_assert(!__is_scoped_enum(IntArNB));
  static_assert(!__is_scoped_enum(HasAnonymousUnion));
  static_assert(!__is_scoped_enum(AnIncompleteType));
  static_assert(!__is_scoped_enum(AnIncompleteTypeAr));
  static_assert(!__is_scoped_enum(AnIncompleteTypeArMB));
  static_assert(!__is_scoped_enum(AnIncompleteTypeArNB));
}

struct FinalClass final {
};

template<typename T>
struct PotentiallyFinal { };

template<typename T>
struct PotentiallyFinal<T*> final { };

template<>
struct PotentiallyFinal<int> final { };

struct FwdDeclFinal;
using FwdDeclFinalAlias = FwdDeclFinal;
struct FwdDeclFinal final {};




void is_final()
{
	static_assert(__is_final(FinalClass));
	static_assert(__is_final(PotentiallyFinal<float*>));
	static_assert(__is_final(PotentiallyFinal<int>));
        static_assert(__is_final(FwdDeclFinal));
        static_assert(__is_final(FwdDeclFinalAlias));

	static_assert(!__is_final(int));
	static_assert(!__is_final(Union));
	static_assert(!__is_final(Int));
	static_assert(!__is_final(IntAr));
	static_assert(!__is_final(UnionAr));
	static_assert(!__is_final(Derives));
	static_assert(!__is_final(ClassType));
	static_assert(!__is_final(cvoid));
	static_assert(!__is_final(IntArNB));
	static_assert(!__is_final(HasAnonymousUnion));
}


typedef HasVirt Polymorph;
struct InheritPolymorph : Polymorph {};

void is_polymorphic()
{
  static_assert(__is_polymorphic(Polymorph));
  static_assert(__is_polymorphic(InheritPolymorph));

  static_assert(!__is_polymorphic(int));
  static_assert(!__is_polymorphic(Union));
  static_assert(!__is_polymorphic(IncompleteUnion));
  static_assert(!__is_polymorphic(Int));
  static_assert(!__is_polymorphic(IntAr));
  static_assert(!__is_polymorphic(UnionAr));
  static_assert(!__is_polymorphic(Derives));
  static_assert(!__is_polymorphic(ClassType));
  static_assert(!__is_polymorphic(Enum));
  static_assert(!__is_polymorphic(cvoid));
  static_assert(!__is_polymorphic(IntArNB));
}

void is_integral()
{
  static_assert(__is_integral(bool));
  static_assert(__is_integral(char));
  static_assert(__is_integral(signed char));
  static_assert(__is_integral(unsigned char));
  //static_assert(__is_integral(char16_t));
  //static_assert(__is_integral(char32_t));
  static_assert(__is_integral(wchar_t));
  static_assert(__is_integral(short));
  static_assert(__is_integral(unsigned short));
  static_assert(__is_integral(int));
  static_assert(__is_integral(unsigned int));
  static_assert(__is_integral(long));
  static_assert(__is_integral(unsigned long));

  static_assert(!__is_integral(float));
  static_assert(!__is_integral(double));
  static_assert(!__is_integral(long double));
  static_assert(!__is_integral(Union));
  static_assert(!__is_integral(UnionAr));
  static_assert(!__is_integral(Derives));
  static_assert(!__is_integral(ClassType));
  static_assert(!__is_integral(Enum));
  static_assert(!__is_integral(void));
  static_assert(!__is_integral(cvoid));
  static_assert(!__is_integral(IntArNB));
}

void is_floating_point()
{
  static_assert(__is_floating_point(float));
  static_assert(__is_floating_point(double));
  static_assert(__is_floating_point(long double));

  static_assert(!__is_floating_point(bool));
  static_assert(!__is_floating_point(char));
  static_assert(!__is_floating_point(signed char));
  static_assert(!__is_floating_point(unsigned char));
  //static_assert(!__is_floating_point(char16_t));
  //static_assert(!__is_floating_point(char32_t));
  static_assert(!__is_floating_point(wchar_t));
  static_assert(!__is_floating_point(short));
  static_assert(!__is_floating_point(unsigned short));
  static_assert(!__is_floating_point(int));
  static_assert(!__is_floating_point(unsigned int));
  static_assert(!__is_floating_point(long));
  static_assert(!__is_floating_point(unsigned long));
  static_assert(!__is_floating_point(Union));
  static_assert(!__is_floating_point(UnionAr));
  static_assert(!__is_floating_point(Derives));
  static_assert(!__is_floating_point(ClassType));
  static_assert(!__is_floating_point(Enum));
  static_assert(!__is_floating_point(void));
  static_assert(!__is_floating_point(cvoid));
  static_assert(!__is_floating_point(IntArNB));
}

template <class T>
struct AggregateTemplate {
  T value;
};

template <class T>
struct NonAggregateTemplate {
  T value;
  NonAggregateTemplate();
};

void is_aggregate()
{
  constexpr bool TrueAfterCpp11 = __cplusplus > 201103L;
  constexpr bool TrueAfterCpp14 = __cplusplus > 201402L;

  __is_aggregate(AnIncompleteType); // expected-error {{incomplete type}}
  __is_aggregate(IncompleteUnion); // expected-error {{incomplete type}}

  // Valid since LWG3823
  static_assert(__is_aggregate(AnIncompleteType[]));
  static_assert(__is_aggregate(AnIncompleteType[1]));
  static_assert(__is_aggregate(AnIncompleteTypeAr));
  static_assert(__is_aggregate(AnIncompleteTypeArNB));
  static_assert(__is_aggregate(AnIncompleteTypeArMB));

  static_assert(!__is_aggregate(NonPOD));
  static_assert(__is_aggregate(NonPODAr));
  static_assert(__is_aggregate(NonPODArNB));
  static_assert(__is_aggregate(NonPODArMB));

  static_assert(!__is_aggregate(Enum));
  static_assert(__is_aggregate(POD));
  static_assert(__is_aggregate(Empty));
  static_assert(__is_aggregate(EmptyAr));
  static_assert(__is_aggregate(EmptyArNB));
  static_assert(__is_aggregate(EmptyArMB));
  static_assert(!__is_aggregate(void));
  static_assert(!__is_aggregate(const volatile void));
  static_assert(!__is_aggregate(int));
  static_assert(__is_aggregate(IntAr));
  static_assert(__is_aggregate(IntArNB));
  static_assert(__is_aggregate(EmptyUnion));
  static_assert(__is_aggregate(Union));
  static_assert(__is_aggregate(Statics));
  static_assert(__is_aggregate(HasFunc));
  static_assert(__is_aggregate(HasOp));
  static_assert(__is_aggregate(HasAssign));
  static_assert(__is_aggregate(HasAnonymousUnion));

  static_assert(__is_aggregate(Derives) == TrueAfterCpp14);
  static_assert(__is_aggregate(DerivesAr));
  static_assert(__is_aggregate(DerivesArNB));
  static_assert(!__is_aggregate(HasCons));
#if __cplusplus >= 202002L
  static_assert(!__is_aggregate(HasDefaultCons));
#else
  static_assert(__is_aggregate(HasDefaultCons));
#endif
  static_assert(!__is_aggregate(HasExplicitDefaultCons));
  static_assert(!__is_aggregate(HasInheritedCons));
  static_assert(__is_aggregate(HasNoInheritedCons) == TrueAfterCpp14);
  static_assert(__is_aggregate(HasCopyAssign));
  static_assert(!__is_aggregate(NonTrivialDefault));
  static_assert(__is_aggregate(HasDest));
  static_assert(!__is_aggregate(HasPriv));
  static_assert(!__is_aggregate(HasProt));
  static_assert(__is_aggregate(HasRefAggregate));
  static_assert(__is_aggregate(HasNonPOD));
  static_assert(!__is_aggregate(HasVirt));
  static_assert(__is_aggregate(VirtAr));
  static_assert(__is_aggregate(HasInClassInit) == TrueAfterCpp11);
  static_assert(!__is_aggregate(HasPrivateBase));
  static_assert(!__is_aggregate(HasProtectedBase));
  static_assert(!__is_aggregate(HasVirtBase));

  static_assert(__is_aggregate(AggregateTemplate<int>));
  static_assert(!__is_aggregate(NonAggregateTemplate<int>));

  static_assert(__is_aggregate(Vector)); // Extension supported by GCC and Clang
  static_assert(__is_aggregate(VectorExt));
  static_assert(__is_aggregate(ComplexInt));
  static_assert(__is_aggregate(ComplexFloat));
}

void is_arithmetic()
{
  static_assert(__is_arithmetic(float));
  static_assert(__is_arithmetic(double));
  static_assert(__is_arithmetic(long double));
  static_assert(__is_arithmetic(bool));
  static_assert(__is_arithmetic(char));
  static_assert(__is_arithmetic(signed char));
  static_assert(__is_arithmetic(unsigned char));
  //static_assert(__is_arithmetic(char16_t));
  //static_assert(__is_arithmetic(char32_t));
  static_assert(__is_arithmetic(wchar_t));
  static_assert(__is_arithmetic(short));
  static_assert(__is_arithmetic(unsigned short));
  static_assert(__is_arithmetic(int));
  static_assert(__is_arithmetic(unsigned int));
  static_assert(__is_arithmetic(long));
  static_assert(__is_arithmetic(unsigned long));

  static_assert(!__is_arithmetic(Union));
  static_assert(!__is_arithmetic(UnionAr));
  static_assert(!__is_arithmetic(Derives));
  static_assert(!__is_arithmetic(ClassType));
  static_assert(!__is_arithmetic(Enum));
  static_assert(!__is_arithmetic(void));
  static_assert(!__is_arithmetic(cvoid));
  static_assert(!__is_arithmetic(IntArNB));
}

void is_complete_type()
{
  static_assert(__is_complete_type(float));
  static_assert(__is_complete_type(double));
  static_assert(__is_complete_type(long double));
  static_assert(__is_complete_type(bool));
  static_assert(__is_complete_type(char));
  static_assert(__is_complete_type(signed char));
  static_assert(__is_complete_type(unsigned char));
  //static_assert(__is_complete_type(char16_t));
  //static_assert(__is_complete_type(char32_t));
  static_assert(__is_complete_type(wchar_t));
  static_assert(__is_complete_type(short));
  static_assert(__is_complete_type(unsigned short));
  static_assert(__is_complete_type(int));
  static_assert(__is_complete_type(unsigned int));
  static_assert(__is_complete_type(long));
  static_assert(__is_complete_type(unsigned long));
  static_assert(__is_complete_type(ACompleteType));

  static_assert(!__is_complete_type(AnIncompleteType));
}

void is_void()
{
  static_assert(__is_void(void));
  static_assert(__is_void(cvoid));

  static_assert(!__is_void(float));
  static_assert(!__is_void(double));
  static_assert(!__is_void(long double));
  static_assert(!__is_void(bool));
  static_assert(!__is_void(char));
  static_assert(!__is_void(signed char));
  static_assert(!__is_void(unsigned char));
  static_assert(!__is_void(wchar_t));
  static_assert(!__is_void(short));
  static_assert(!__is_void(unsigned short));
  static_assert(!__is_void(int));
  static_assert(!__is_void(unsigned int));
  static_assert(!__is_void(long));
  static_assert(!__is_void(unsigned long));
  static_assert(!__is_void(Union));
  static_assert(!__is_void(UnionAr));
  static_assert(!__is_void(Derives));
  static_assert(!__is_void(ClassType));
  static_assert(!__is_void(Enum));
  static_assert(!__is_void(IntArNB));
  static_assert(!__is_void(void*));
  static_assert(!__is_void(cvoid*));
}

void is_array()
{
  static_assert(__is_array(IntAr));
  static_assert(__is_array(IntArNB));
  static_assert(!__is_array(IntArZero));
  static_assert(__is_array(UnionAr));

  static_assert(!__is_array(void));
  static_assert(!__is_array(cvoid));
  static_assert(!__is_array(float));
  static_assert(!__is_array(double));
  static_assert(!__is_array(long double));
  static_assert(!__is_array(bool));
  static_assert(!__is_array(char));
  static_assert(!__is_array(signed char));
  static_assert(!__is_array(unsigned char));
  static_assert(!__is_array(wchar_t));
  static_assert(!__is_array(short));
  static_assert(!__is_array(unsigned short));
  static_assert(!__is_array(int));
  static_assert(!__is_array(unsigned int));
  static_assert(!__is_array(long));
  static_assert(!__is_array(unsigned long));
  static_assert(!__is_array(Union));
  static_assert(!__is_array(Derives));
  static_assert(!__is_array(ClassType));
  static_assert(!__is_array(Enum));
  static_assert(!__is_array(void*));
  static_assert(!__is_array(cvoid*));
}

void is_bounded_array(int n) {
  static_assert(__is_bounded_array(IntAr));
  static_assert(!__is_bounded_array(IntArNB));
  static_assert(!__is_bounded_array(IntArZero));
  static_assert(__is_bounded_array(UnionAr));

  static_assert(!__is_bounded_array(void));
  static_assert(!__is_bounded_array(cvoid));
  static_assert(!__is_bounded_array(float));
  static_assert(!__is_bounded_array(double));
  static_assert(!__is_bounded_array(long double));
  static_assert(!__is_bounded_array(bool));
  static_assert(!__is_bounded_array(char));
  static_assert(!__is_bounded_array(signed char));
  static_assert(!__is_bounded_array(unsigned char));
  static_assert(!__is_bounded_array(wchar_t));
  static_assert(!__is_bounded_array(short));
  static_assert(!__is_bounded_array(unsigned short));
  static_assert(!__is_bounded_array(int));
  static_assert(!__is_bounded_array(unsigned int));
  static_assert(!__is_bounded_array(long));
  static_assert(!__is_bounded_array(unsigned long));
  static_assert(!__is_bounded_array(Union));
  static_assert(!__is_bounded_array(Derives));
  static_assert(!__is_bounded_array(ClassType));
  static_assert(!__is_bounded_array(Enum));
  static_assert(!__is_bounded_array(void *));
  static_assert(!__is_bounded_array(cvoid *));

  int t32[n];
  (void)__is_bounded_array(decltype(t32)); // expected-error{{variable length arrays are not supported in '__is_bounded_array'}}
}

void is_unbounded_array(int n) {
  static_assert(!__is_unbounded_array(IntAr));
  static_assert(__is_unbounded_array(IntArNB));
  static_assert(!__is_unbounded_array(IntArZero));
  static_assert(!__is_unbounded_array(UnionAr));

  static_assert(!__is_unbounded_array(void));
  static_assert(!__is_unbounded_array(cvoid));
  static_assert(!__is_unbounded_array(float));
  static_assert(!__is_unbounded_array(double));
  static_assert(!__is_unbounded_array(long double));
  static_assert(!__is_unbounded_array(bool));
  static_assert(!__is_unbounded_array(char));
  static_assert(!__is_unbounded_array(signed char));
  static_assert(!__is_unbounded_array(unsigned char));
  static_assert(!__is_unbounded_array(wchar_t));
  static_assert(!__is_unbounded_array(short));
  static_assert(!__is_unbounded_array(unsigned short));
  static_assert(!__is_unbounded_array(int));
  static_assert(!__is_unbounded_array(unsigned int));
  static_assert(!__is_unbounded_array(long));
  static_assert(!__is_unbounded_array(unsigned long));
  static_assert(!__is_unbounded_array(Union));
  static_assert(!__is_unbounded_array(Derives));
  static_assert(!__is_unbounded_array(ClassType));
  static_assert(!__is_unbounded_array(Enum));
  static_assert(!__is_unbounded_array(void *));
  static_assert(!__is_unbounded_array(cvoid *));

  int t32[n];
  (void)__is_unbounded_array(decltype(t32)); // expected-error{{variable length arrays are not supported in '__is_unbounded_array'}}
}

template <typename T> void tmpl_func(T&) {}

template <typename T> struct type_wrapper {
  typedef T type;
  typedef T* ptrtype;
  typedef T& reftype;
};

void is_function()
{
  static_assert(__is_function(type_wrapper<void(void)>::type));
  static_assert(__is_function(typeof(tmpl_func<int>)));

  typedef void (*ptr_to_func_type)(void);

  static_assert(!__is_function(void));
  static_assert(!__is_function(cvoid));
  static_assert(!__is_function(float));
  static_assert(!__is_function(double));
  static_assert(!__is_function(long double));
  static_assert(!__is_function(bool));
  static_assert(!__is_function(char));
  static_assert(!__is_function(signed char));
  static_assert(!__is_function(unsigned char));
  static_assert(!__is_function(wchar_t));
  static_assert(!__is_function(short));
  static_assert(!__is_function(unsigned short));
  static_assert(!__is_function(int));
  static_assert(!__is_function(unsigned int));
  static_assert(!__is_function(long));
  static_assert(!__is_function(unsigned long));
  static_assert(!__is_function(Union));
  static_assert(!__is_function(Derives));
  static_assert(!__is_function(ClassType));
  static_assert(!__is_function(Enum));
  static_assert(!__is_function(void*));
  static_assert(!__is_function(cvoid*));
  static_assert(!__is_function(void(*)()));
  static_assert(!__is_function(ptr_to_func_type));
  static_assert(!__is_function(type_wrapper<void(void)>::ptrtype));
  static_assert(!__is_function(type_wrapper<void(void)>::reftype));
}

void is_reference()
{
  static_assert(__is_reference(int&));
  static_assert(__is_reference(const int&));
  static_assert(__is_reference(void *&));

  static_assert(!__is_reference(int));
  static_assert(!__is_reference(const int));
  static_assert(!__is_reference(void *));
}

void is_lvalue_reference()
{
  static_assert(__is_lvalue_reference(int&));
  static_assert(__is_lvalue_reference(void *&));
  static_assert(__is_lvalue_reference(const int&));
  static_assert(__is_lvalue_reference(void * const &));

  static_assert(!__is_lvalue_reference(int));
  static_assert(!__is_lvalue_reference(const int));
  static_assert(!__is_lvalue_reference(void *));
}

#if __has_feature(cxx_rvalue_references)

void is_rvalue_reference()
{
  static_assert(__is_rvalue_reference(const int&&));
  static_assert(__is_rvalue_reference(void * const &&));

  static_assert(!__is_rvalue_reference(int&));
  static_assert(!__is_rvalue_reference(void *&));
  static_assert(!__is_rvalue_reference(const int&));
  static_assert(!__is_rvalue_reference(void * const &));
  static_assert(!__is_rvalue_reference(int));
  static_assert(!__is_rvalue_reference(const int));
  static_assert(!__is_rvalue_reference(void *));
}

#endif

void is_fundamental()
{
  static_assert(__is_fundamental(float));
  static_assert(__is_fundamental(double));
  static_assert(__is_fundamental(long double));
  static_assert(__is_fundamental(bool));
  static_assert(__is_fundamental(char));
  static_assert(__is_fundamental(signed char));
  static_assert(__is_fundamental(unsigned char));
  //static_assert(__is_fundamental(char16_t));
  //static_assert(__is_fundamental(char32_t));
  static_assert(__is_fundamental(wchar_t));
  static_assert(__is_fundamental(short));
  static_assert(__is_fundamental(unsigned short));
  static_assert(__is_fundamental(int));
  static_assert(__is_fundamental(unsigned int));
  static_assert(__is_fundamental(long));
  static_assert(__is_fundamental(unsigned long));
  static_assert(__is_fundamental(void));
  static_assert(__is_fundamental(cvoid));
  static_assert(__is_fundamental(decltype(nullptr)));

  static_assert(!__is_fundamental(Union));
  static_assert(!__is_fundamental(UnionAr));
  static_assert(!__is_fundamental(Derives));
  static_assert(!__is_fundamental(ClassType));
  static_assert(!__is_fundamental(Enum));
  static_assert(!__is_fundamental(IntArNB));
}

void is_object()
{
  static_assert(__is_object(int));
  static_assert(__is_object(int *));
  static_assert(__is_object(void *));
  static_assert(__is_object(Union));
  static_assert(__is_object(UnionAr));
  static_assert(__is_object(ClassType));
  static_assert(__is_object(Enum));

  static_assert(!__is_object(type_wrapper<void(void)>::type));
  static_assert(!__is_object(int&));
  static_assert(!__is_object(void));
}

void is_scalar()
{
  static_assert(__is_scalar(float));
  static_assert(__is_scalar(double));
  static_assert(__is_scalar(long double));
  static_assert(__is_scalar(bool));
  static_assert(__is_scalar(char));
  static_assert(__is_scalar(signed char));
  static_assert(__is_scalar(unsigned char));
  static_assert(__is_scalar(wchar_t));
  static_assert(__is_scalar(short));
  static_assert(__is_scalar(unsigned short));
  static_assert(__is_scalar(int));
  static_assert(__is_scalar(unsigned int));
  static_assert(__is_scalar(long));
  static_assert(__is_scalar(unsigned long));
  static_assert(__is_scalar(Enum));
  static_assert(__is_scalar(void*));
  static_assert(__is_scalar(cvoid*));

  static_assert(!__is_scalar(void));
  static_assert(!__is_scalar(cvoid));
  static_assert(!__is_scalar(Union));
  static_assert(!__is_scalar(UnionAr));
  static_assert(!__is_scalar(Derives));
  static_assert(!__is_scalar(ClassType));
  static_assert(!__is_scalar(IntArNB));
}

struct StructWithMembers {
  int member;
  void method() {}
};

void is_compound()
{
  static_assert(__is_compound(void*));
  static_assert(__is_compound(cvoid*));
  static_assert(__is_compound(void (*)()));
  static_assert(__is_compound(int StructWithMembers::*));
  static_assert(__is_compound(void (StructWithMembers::*)()));
  static_assert(__is_compound(int&));
  static_assert(__is_compound(Union));
  static_assert(__is_compound(UnionAr));
  static_assert(__is_compound(Derives));
  static_assert(__is_compound(ClassType));
  static_assert(__is_compound(IntArNB));
  static_assert(__is_compound(Enum));

  static_assert(!__is_compound(float));
  static_assert(!__is_compound(double));
  static_assert(!__is_compound(long double));
  static_assert(!__is_compound(bool));
  static_assert(!__is_compound(char));
  static_assert(!__is_compound(signed char));
  static_assert(!__is_compound(unsigned char));
  static_assert(!__is_compound(wchar_t));
  static_assert(!__is_compound(short));
  static_assert(!__is_compound(unsigned short));
  static_assert(!__is_compound(int));
  static_assert(!__is_compound(unsigned int));
  static_assert(!__is_compound(long));
  static_assert(!__is_compound(unsigned long));
  static_assert(!__is_compound(void));
  static_assert(!__is_compound(cvoid));
}

void is_pointer()
{
  StructWithMembers x;

  static_assert(__is_pointer(void*));
  static_assert(__is_pointer(cvoid*));
  static_assert(__is_pointer(cvoid*));
  static_assert(__is_pointer(char*));
  static_assert(__is_pointer(int*));
  static_assert(__is_pointer(int**));
  static_assert(__is_pointer(ClassType*));
  static_assert(__is_pointer(Derives*));
  static_assert(__is_pointer(Enum*));
  static_assert(__is_pointer(IntArNB*));
  static_assert(__is_pointer(Union*));
  static_assert(__is_pointer(UnionAr*));
  static_assert(__is_pointer(StructWithMembers*));
  static_assert(__is_pointer(void (*)()));

  static_assert(!__is_pointer(void));
  static_assert(!__is_pointer(cvoid));
  static_assert(!__is_pointer(cvoid));
  static_assert(!__is_pointer(char));
  static_assert(!__is_pointer(int));
  static_assert(!__is_pointer(int));
  static_assert(!__is_pointer(ClassType));
  static_assert(!__is_pointer(Derives));
  static_assert(!__is_pointer(Enum));
  static_assert(!__is_pointer(IntArNB));
  static_assert(!__is_pointer(Union));
  static_assert(!__is_pointer(UnionAr));
  static_assert(!__is_pointer(StructWithMembers));
  static_assert(!__is_pointer(int StructWithMembers::*));
  static_assert(!__is_pointer(void (StructWithMembers::*) ()));
}

void is_member_object_pointer()
{
  StructWithMembers x;

  static_assert(__is_member_object_pointer(int StructWithMembers::*));

  static_assert(!__is_member_object_pointer(void (StructWithMembers::*) ()));
  static_assert(!__is_member_object_pointer(void*));
  static_assert(!__is_member_object_pointer(cvoid*));
  static_assert(!__is_member_object_pointer(cvoid*));
  static_assert(!__is_member_object_pointer(char*));
  static_assert(!__is_member_object_pointer(int*));
  static_assert(!__is_member_object_pointer(int**));
  static_assert(!__is_member_object_pointer(ClassType*));
  static_assert(!__is_member_object_pointer(Derives*));
  static_assert(!__is_member_object_pointer(Enum*));
  static_assert(!__is_member_object_pointer(IntArNB*));
  static_assert(!__is_member_object_pointer(Union*));
  static_assert(!__is_member_object_pointer(UnionAr*));
  static_assert(!__is_member_object_pointer(StructWithMembers*));
  static_assert(!__is_member_object_pointer(void));
  static_assert(!__is_member_object_pointer(cvoid));
  static_assert(!__is_member_object_pointer(cvoid));
  static_assert(!__is_member_object_pointer(char));
  static_assert(!__is_member_object_pointer(int));
  static_assert(!__is_member_object_pointer(int));
  static_assert(!__is_member_object_pointer(ClassType));
  static_assert(!__is_member_object_pointer(Derives));
  static_assert(!__is_member_object_pointer(Enum));
  static_assert(!__is_member_object_pointer(IntArNB));
  static_assert(!__is_member_object_pointer(Union));
  static_assert(!__is_member_object_pointer(UnionAr));
  static_assert(!__is_member_object_pointer(StructWithMembers));
  static_assert(!__is_member_object_pointer(void (*)()));
}

void is_member_function_pointer()
{
  StructWithMembers x;

  static_assert(__is_member_function_pointer(void (StructWithMembers::*) ()));

  static_assert(!__is_member_function_pointer(int StructWithMembers::*));
  static_assert(!__is_member_function_pointer(void*));
  static_assert(!__is_member_function_pointer(cvoid*));
  static_assert(!__is_member_function_pointer(cvoid*));
  static_assert(!__is_member_function_pointer(char*));
  static_assert(!__is_member_function_pointer(int*));
  static_assert(!__is_member_function_pointer(int**));
  static_assert(!__is_member_function_pointer(ClassType*));
  static_assert(!__is_member_function_pointer(Derives*));
  static_assert(!__is_member_function_pointer(Enum*));
  static_assert(!__is_member_function_pointer(IntArNB*));
  static_assert(!__is_member_function_pointer(Union*));
  static_assert(!__is_member_function_pointer(UnionAr*));
  static_assert(!__is_member_function_pointer(StructWithMembers*));
  static_assert(!__is_member_function_pointer(void));
  static_assert(!__is_member_function_pointer(cvoid));
  static_assert(!__is_member_function_pointer(cvoid));
  static_assert(!__is_member_function_pointer(char));
  static_assert(!__is_member_function_pointer(int));
  static_assert(!__is_member_function_pointer(int));
  static_assert(!__is_member_function_pointer(ClassType));
  static_assert(!__is_member_function_pointer(Derives));
  static_assert(!__is_member_function_pointer(Enum));
  static_assert(!__is_member_function_pointer(IntArNB));
  static_assert(!__is_member_function_pointer(Union));
  static_assert(!__is_member_function_pointer(UnionAr));
  static_assert(!__is_member_function_pointer(StructWithMembers));
  static_assert(!__is_member_function_pointer(void (*)()));
}

void is_member_pointer()
{
  StructWithMembers x;

  static_assert(__is_member_pointer(int StructWithMembers::*));
  static_assert(__is_member_pointer(void (StructWithMembers::*) ()));

  static_assert(!__is_member_pointer(void*));
  static_assert(!__is_member_pointer(cvoid*));
  static_assert(!__is_member_pointer(cvoid*));
  static_assert(!__is_member_pointer(char*));
  static_assert(!__is_member_pointer(int*));
  static_assert(!__is_member_pointer(int**));
  static_assert(!__is_member_pointer(ClassType*));
  static_assert(!__is_member_pointer(Derives*));
  static_assert(!__is_member_pointer(Enum*));
  static_assert(!__is_member_pointer(IntArNB*));
  static_assert(!__is_member_pointer(Union*));
  static_assert(!__is_member_pointer(UnionAr*));
  static_assert(!__is_member_pointer(StructWithMembers*));
  static_assert(!__is_member_pointer(void));
  static_assert(!__is_member_pointer(cvoid));
  static_assert(!__is_member_pointer(cvoid));
  static_assert(!__is_member_pointer(char));
  static_assert(!__is_member_pointer(int));
  static_assert(!__is_member_pointer(int));
  static_assert(!__is_member_pointer(ClassType));
  static_assert(!__is_member_pointer(Derives));
  static_assert(!__is_member_pointer(Enum));
  static_assert(!__is_member_pointer(IntArNB));
  static_assert(!__is_member_pointer(Union));
  static_assert(!__is_member_pointer(UnionAr));
  static_assert(!__is_member_pointer(StructWithMembers));
  static_assert(!__is_member_pointer(void (*)()));
}

void is_const()
{
  static_assert(__is_const(cvoid));
  static_assert(__is_const(const char));
  static_assert(__is_const(const int));
  static_assert(__is_const(const long));
  static_assert(__is_const(const short));
  static_assert(__is_const(const signed char));
  static_assert(__is_const(const wchar_t));
  static_assert(__is_const(const bool));
  static_assert(__is_const(const float));
  static_assert(__is_const(const double));
  static_assert(__is_const(const long double));
  static_assert(__is_const(const unsigned char));
  static_assert(__is_const(const unsigned int));
  static_assert(__is_const(const unsigned long long));
  static_assert(__is_const(const unsigned long));
  static_assert(__is_const(const unsigned short));
  static_assert(__is_const(const void));
  static_assert(__is_const(const ClassType));
  static_assert(__is_const(const Derives));
  static_assert(__is_const(const Enum));
  static_assert(__is_const(const IntArNB));
  static_assert(__is_const(const Union));
  static_assert(__is_const(const UnionAr));

  static_assert(!__is_const(char));
  static_assert(!__is_const(int));
  static_assert(!__is_const(long));
  static_assert(!__is_const(short));
  static_assert(!__is_const(signed char));
  static_assert(!__is_const(wchar_t));
  static_assert(!__is_const(bool));
  static_assert(!__is_const(float));
  static_assert(!__is_const(double));
  static_assert(!__is_const(long double));
  static_assert(!__is_const(unsigned char));
  static_assert(!__is_const(unsigned int));
  static_assert(!__is_const(unsigned long long));
  static_assert(!__is_const(unsigned long));
  static_assert(!__is_const(unsigned short));
  static_assert(!__is_const(void));
  static_assert(!__is_const(ClassType));
  static_assert(!__is_const(Derives));
  static_assert(!__is_const(Enum));
  static_assert(!__is_const(IntArNB));
  static_assert(!__is_const(Union));
  static_assert(!__is_const(UnionAr));
}

void is_volatile()
{
  static_assert(__is_volatile(volatile char));
  static_assert(__is_volatile(volatile int));
  static_assert(__is_volatile(volatile long));
  static_assert(__is_volatile(volatile short));
  static_assert(__is_volatile(volatile signed char));
  static_assert(__is_volatile(volatile wchar_t));
  static_assert(__is_volatile(volatile bool));
  static_assert(__is_volatile(volatile float));
  static_assert(__is_volatile(volatile double));
  static_assert(__is_volatile(volatile long double));
  static_assert(__is_volatile(volatile unsigned char));
  static_assert(__is_volatile(volatile unsigned int));
  static_assert(__is_volatile(volatile unsigned long long));
  static_assert(__is_volatile(volatile unsigned long));
  static_assert(__is_volatile(volatile unsigned short));
  static_assert(__is_volatile(volatile void));
  static_assert(__is_volatile(volatile ClassType));
  static_assert(__is_volatile(volatile Derives));
  static_assert(__is_volatile(volatile Enum));
  static_assert(__is_volatile(volatile IntArNB));
  static_assert(__is_volatile(volatile Union));
  static_assert(__is_volatile(volatile UnionAr));

  static_assert(!__is_volatile(char));
  static_assert(!__is_volatile(int));
  static_assert(!__is_volatile(long));
  static_assert(!__is_volatile(short));
  static_assert(!__is_volatile(signed char));
  static_assert(!__is_volatile(wchar_t));
  static_assert(!__is_volatile(bool));
  static_assert(!__is_volatile(float));
  static_assert(!__is_volatile(double));
  static_assert(!__is_volatile(long double));
  static_assert(!__is_volatile(unsigned char));
  static_assert(!__is_volatile(unsigned int));
  static_assert(!__is_volatile(unsigned long long));
  static_assert(!__is_volatile(unsigned long));
  static_assert(!__is_volatile(unsigned short));
  static_assert(!__is_volatile(void));
  static_assert(!__is_volatile(ClassType));
  static_assert(!__is_volatile(Derives));
  static_assert(!__is_volatile(Enum));
  static_assert(!__is_volatile(IntArNB));
  static_assert(!__is_volatile(Union));
  static_assert(!__is_volatile(UnionAr));
}

struct TrivialStruct {
  int member;
};

struct NonTrivialStruct {
  int member;
  NonTrivialStruct() {
    member = 0;
  }
};

struct SuperNonTrivialStruct {
  SuperNonTrivialStruct() { }
  ~SuperNonTrivialStruct() { }
};

struct NonTCStruct {
  NonTCStruct(const NonTCStruct&) {}
};

struct AllDefaulted {
  AllDefaulted() = default;
  AllDefaulted(const AllDefaulted &) = default;
  AllDefaulted(AllDefaulted &&) = default;
  AllDefaulted &operator=(const AllDefaulted &) = default;
  AllDefaulted &operator=(AllDefaulted &&) = default;
  ~AllDefaulted() = default;
};

struct NoDefaultMoveAssignDueToUDCopyCtor {
  NoDefaultMoveAssignDueToUDCopyCtor(const NoDefaultMoveAssignDueToUDCopyCtor&);
};

struct NoDefaultMoveAssignDueToUDCopyAssign {
  NoDefaultMoveAssignDueToUDCopyAssign& operator=(
    const NoDefaultMoveAssignDueToUDCopyAssign&);
};

struct NoDefaultMoveAssignDueToDtor {
  ~NoDefaultMoveAssignDueToDtor();
};

struct AllDeleted {
  AllDeleted() = delete;
  AllDeleted(const AllDeleted &) = delete;
  AllDeleted(AllDeleted &&) = delete;
  AllDeleted &operator=(const AllDeleted &) = delete;
  AllDeleted &operator=(AllDeleted &&) = delete;
  ~AllDeleted() = delete;
};

struct ExtDefaulted {
  ExtDefaulted();
  ExtDefaulted(const ExtDefaulted &);
  ExtDefaulted(ExtDefaulted &&);
  ExtDefaulted &operator=(const ExtDefaulted &);
  ExtDefaulted &operator=(ExtDefaulted &&);
  ~ExtDefaulted();
};

// Despite being defaulted, these functions are not trivial.
ExtDefaulted::ExtDefaulted() = default;
ExtDefaulted::ExtDefaulted(const ExtDefaulted &) = default;
ExtDefaulted::ExtDefaulted(ExtDefaulted &&) = default;
ExtDefaulted &ExtDefaulted::operator=(const ExtDefaulted &) = default;
ExtDefaulted &ExtDefaulted::operator=(ExtDefaulted &&) = default;
ExtDefaulted::~ExtDefaulted() = default;

void is_trivial2()
{
  static_assert(__is_trivial(char));
  static_assert(__is_trivial(int));
  static_assert(__is_trivial(long));
  static_assert(__is_trivial(short));
  static_assert(__is_trivial(signed char));
  static_assert(__is_trivial(wchar_t));
  static_assert(__is_trivial(bool));
  static_assert(__is_trivial(float));
  static_assert(__is_trivial(double));
  static_assert(__is_trivial(long double));
  static_assert(__is_trivial(unsigned char));
  static_assert(__is_trivial(unsigned int));
  static_assert(__is_trivial(unsigned long long));
  static_assert(__is_trivial(unsigned long));
  static_assert(__is_trivial(unsigned short));
  static_assert(__is_trivial(ClassType));
  static_assert(__is_trivial(Derives));
  static_assert(__is_trivial(Enum));
  static_assert(__is_trivial(IntAr));
  static_assert(__is_trivial(Union));
  static_assert(__is_trivial(UnionAr));
  static_assert(__is_trivial(TrivialStruct));
  static_assert(__is_trivial(AllDefaulted));
  static_assert(__is_trivial(AllDeleted));

  static_assert(!__is_trivial(void));
  static_assert(!__is_trivial(NonTrivialStruct));
  static_assert(!__is_trivial(SuperNonTrivialStruct));
  static_assert(!__is_trivial(NonTCStruct));
  static_assert(!__is_trivial(ExtDefaulted));

  static_assert(__is_trivial(ACompleteType));
  static_assert(!__is_trivial(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_trivial(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__is_trivial(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_trivial(void));
  static_assert(!__is_trivial(const volatile void));
}

void is_trivially_copyable2()
{
  static_assert(__is_trivially_copyable(char));
  static_assert(__is_trivially_copyable(int));
  static_assert(__is_trivially_copyable(long));
  static_assert(__is_trivially_copyable(short));
  static_assert(__is_trivially_copyable(signed char));
  static_assert(__is_trivially_copyable(wchar_t));
  static_assert(__is_trivially_copyable(bool));
  static_assert(__is_trivially_copyable(float));
  static_assert(__is_trivially_copyable(double));
  static_assert(__is_trivially_copyable(long double));
  static_assert(__is_trivially_copyable(unsigned char));
  static_assert(__is_trivially_copyable(unsigned int));
  static_assert(__is_trivially_copyable(unsigned long long));
  static_assert(__is_trivially_copyable(unsigned long));
  static_assert(__is_trivially_copyable(unsigned short));
  static_assert(__is_trivially_copyable(ClassType));
  static_assert(__is_trivially_copyable(Derives));
  static_assert(__is_trivially_copyable(Enum));
  static_assert(__is_trivially_copyable(IntAr));
  static_assert(__is_trivially_copyable(Union));
  static_assert(__is_trivially_copyable(UnionAr));
  static_assert(__is_trivially_copyable(TrivialStruct));
  static_assert(__is_trivially_copyable(NonTrivialStruct));
  static_assert(__is_trivially_copyable(AllDefaulted));
  static_assert(__is_trivially_copyable(AllDeleted));

  static_assert(!__is_trivially_copyable(void));
  static_assert(!__is_trivially_copyable(SuperNonTrivialStruct));
  static_assert(!__is_trivially_copyable(NonTCStruct));
  static_assert(!__is_trivially_copyable(ExtDefaulted));

  static_assert(__is_trivially_copyable(const int));
  static_assert(__is_trivially_copyable(volatile int));

  static_assert(__is_trivially_copyable(ACompleteType));
  static_assert(!__is_trivially_copyable(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_copyable(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_copyable(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_copyable(void));
  static_assert(!__is_trivially_copyable(const volatile void));
}

struct CStruct {
  int one;
  int two;
};

struct CEmptyStruct {};

struct CppEmptyStruct : CStruct {};
struct CppStructStandard : CEmptyStruct {
  int three;
  int four;
};
struct CppStructNonStandardByBase : CStruct {
  int three;
  int four;
};
struct CppStructNonStandardByVirt : CStruct {
  virtual void method() {}
};
struct CppStructNonStandardByMemb : CStruct {
  CppStructNonStandardByVirt member;
};
struct CppStructNonStandardByProt : CStruct {
  int five;
protected:
  int six;
};
struct CppStructNonStandardByVirtBase : virtual CStruct {
};
struct CppStructNonStandardBySameBase : CEmptyStruct {
  CEmptyStruct member;
};
struct CppStructNonStandardBy2ndVirtBase : CEmptyStruct {
  CEmptyStruct member;
};

void is_standard_layout()
{
  typedef const int ConstInt;
  typedef ConstInt ConstIntAr[4];
  typedef CppStructStandard CppStructStandardAr[4];

  static_assert(__is_standard_layout(int));
  static_assert(__is_standard_layout(ConstInt));
  static_assert(__is_standard_layout(ConstIntAr));
  static_assert(__is_standard_layout(CStruct));
  static_assert(__is_standard_layout(CppStructStandard));
  static_assert(__is_standard_layout(CppStructStandardAr));
  static_assert(__is_standard_layout(Vector));
  static_assert(__is_standard_layout(VectorExt));

  typedef CppStructNonStandardByBase CppStructNonStandardByBaseAr[4];

  static_assert(!__is_standard_layout(CppStructNonStandardByVirt));
  static_assert(!__is_standard_layout(CppStructNonStandardByMemb));
  static_assert(!__is_standard_layout(CppStructNonStandardByProt));
  static_assert(!__is_standard_layout(CppStructNonStandardByVirtBase));
  static_assert(!__is_standard_layout(CppStructNonStandardByBase));
  static_assert(!__is_standard_layout(CppStructNonStandardByBaseAr));
  static_assert(!__is_standard_layout(CppStructNonStandardBySameBase));
  static_assert(!__is_standard_layout(CppStructNonStandardBy2ndVirtBase));

  static_assert(__is_standard_layout(ACompleteType));
  static_assert(!__is_standard_layout(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_standard_layout(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__is_standard_layout(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_standard_layout(void));
  static_assert(!__is_standard_layout(const volatile void));

  struct HasAnonEmptyBitfield { int : 0; };
  struct HasAnonBitfield { int : 4; };
  struct DerivesFromBitfield : HasAnonBitfield {};
  struct DerivesFromBitfieldWithBitfield : HasAnonBitfield { int : 5; };
  struct DerivesFromBitfieldTwice : DerivesFromBitfield, HasAnonEmptyBitfield {};

  static_assert(__is_standard_layout(HasAnonEmptyBitfield));
  static_assert(__is_standard_layout(HasAnonBitfield));
  static_assert(__is_standard_layout(DerivesFromBitfield));
  static_assert(!__is_standard_layout(DerivesFromBitfieldWithBitfield));
  static_assert(!__is_standard_layout(DerivesFromBitfieldTwice));

  struct Empty {};
  struct HasEmptyBase : Empty {};
  struct HoldsEmptyBase { Empty e; };
  struct HasRepeatedEmptyBase : Empty, HasEmptyBase {}; // expected-warning {{inaccessible}}
  struct HasEmptyBaseAsMember : Empty { Empty e; };
  struct HasEmptyBaseAsSubobjectOfMember1 : Empty { HoldsEmptyBase e; };
  struct HasEmptyBaseAsSubobjectOfMember2 : Empty { HasEmptyBase e; };
  struct HasEmptyBaseAsSubobjectOfMember3 : Empty { HoldsEmptyBase e[2]; };
  struct HasEmptyIndirectBaseAsMember : HasEmptyBase { Empty e; };
  struct HasEmptyIndirectBaseAsSecondMember : HasEmptyBase { int n; Empty e; };
  struct HasEmptyIndirectBaseAfterBitfield : HasEmptyBase { int : 4; Empty e; };

  static_assert(__is_standard_layout(Empty));
  static_assert(__is_standard_layout(HasEmptyBase));
  static_assert(!__is_standard_layout(HasRepeatedEmptyBase));
  static_assert(!__is_standard_layout(HasEmptyBaseAsMember));
  static_assert(!__is_standard_layout(HasEmptyBaseAsSubobjectOfMember1));
  static_assert(__is_standard_layout(HasEmptyBaseAsSubobjectOfMember2)); // FIXME: standard bug?
  static_assert(!__is_standard_layout(HasEmptyBaseAsSubobjectOfMember3));
  static_assert(!__is_standard_layout(HasEmptyIndirectBaseAsMember));
  static_assert(__is_standard_layout(HasEmptyIndirectBaseAsSecondMember));
  static_assert(!__is_standard_layout(HasEmptyIndirectBaseAfterBitfield)); // FIXME: standard bug?

  struct StructWithEmptyFields {
    int n;
    HoldsEmptyBase e[3];
  };
  union UnionWithEmptyFields {
    int n;
    HoldsEmptyBase e[3];
  };
  struct HasEmptyIndirectBaseAsSecondStructMember : HasEmptyBase {
    StructWithEmptyFields u;
  };
  struct HasEmptyIndirectBaseAsSecondUnionMember : HasEmptyBase {
    UnionWithEmptyFields u;
  };

  static_assert(__is_standard_layout(HasEmptyIndirectBaseAsSecondStructMember));
  static_assert(!__is_standard_layout(HasEmptyIndirectBaseAsSecondUnionMember));
}

struct CStruct2 {
  int one;
  int two;
};

struct CEmptyStruct2 {};

struct CppEmptyStruct2 : CStruct2 {};
struct CppStructStandard2 : CEmptyStruct2 {
  int three;
  int four;
};
struct CppStructNonStandardByBase2 : CStruct2 {
  int three;
  int four;
};
struct CppStructNonStandardByVirt2 : CStruct2 {
  virtual void method() {}
};
struct CppStructNonStandardByMemb2 : CStruct2 {
  CppStructNonStandardByVirt member;
};
struct CppStructNonStandardByProt2 : CStruct2 {
  int five;
protected:
  int six;
};
struct CppStructNonStandardByVirtBase2 : virtual CStruct2 {
};
struct CppStructNonStandardBySameBase2 : CEmptyStruct2 {
  CEmptyStruct member;
};
struct CppStructNonStandardBy2ndVirtBase2 : CEmptyStruct2 {
  CEmptyStruct member;
};

struct CStructWithQualifiers {
  const int one;
  volatile int two;
};

struct CStructNoUniqueAddress {
  int one;
  [[no_unique_address]] int two;
};

struct CStructNoUniqueAddress2 {
  int one;
  [[no_unique_address]] int two;
};

struct alignas(64) CStructAlignment {
  int one;
  int two;
};

struct CStructAlignedMembers {
  int one;
  alignas(16) int two;
};

enum EnumLayout : int {};
enum class EnumClassLayout {};
enum EnumForward : int;
enum class EnumClassForward;

struct CStructIncomplete; // #CStructIncomplete

struct CStructNested {
  int a;
  CStruct s;
  int b;
};

struct CStructNested2 {
  int a2;
  CStruct s2;
  int b2;
};

struct CStructWithBitfelds {
  int a : 5;
  int : 0;
};

struct CStructWithBitfelds2 {
  int a : 5;
  int : 0;
};

struct CStructWithBitfelds3 {
  int : 0;
  int b : 5;
};

struct CStructWithBitfelds4 {
  EnumLayout a : 5;
  int : 0;
};

union UnionLayout {
  int a;
  double b;
  CStruct c;
  [[no_unique_address]] CEmptyStruct d;
  [[no_unique_address]] CEmptyStruct2 e;
};

union UnionLayout2 {
  CStruct c;
  int a;
  CEmptyStruct2 e;
  double b;
  [[no_unique_address]] CEmptyStruct d;
};

union UnionLayout3 {
  CStruct c;
  int a;
  double b;
  [[no_unique_address]] CEmptyStruct d;
};

union UnionNoOveralignedMembers {
  int a;
  double b;
};

union UnionWithOveralignedMembers {
  int a;
  alignas(16) double b;
};

struct StructWithAnonUnion {
  union {
    int a;
    double b;
    CStruct c;
    [[no_unique_address]] CEmptyStruct d;
    [[no_unique_address]] CEmptyStruct2 e;
  };
};

struct StructWithAnonUnion2 {
  union {
    CStruct c;
    int a;
    CEmptyStruct2 e;
    double b;
    [[no_unique_address]] CEmptyStruct d;
  };
};

struct StructWithAnonUnion3 {
  union {
    CStruct c;
    int a;
    CEmptyStruct2 e;
    double b;
    [[no_unique_address]] CEmptyStruct d;
  } u;
};

struct CStructWithArrayAtTheEnd {
  int a;
  int b[4];
};

struct CStructWithFMA {
  int c;
  int d[];
};

struct CStructWithFMA2 {
  int e;
  int f[];
};

template<int N>
struct UniqueEmpty {};
template<typename... Bases>
struct D : Bases... {};

void is_layout_compatible(int n)
{
  static_assert(__is_layout_compatible(void, void));
  static_assert(!__is_layout_compatible(void, int));
  static_assert(__is_layout_compatible(void, const void));
  static_assert(__is_layout_compatible(void, volatile void));
  static_assert(__is_layout_compatible(const void, volatile void));
  static_assert(__is_layout_compatible(int, int));
  static_assert(__is_layout_compatible(int, const int));
  static_assert(__is_layout_compatible(int, volatile int));
  static_assert(__is_layout_compatible(const int, volatile int));
  static_assert(__is_layout_compatible(int *, int * __restrict));
  // Note: atomic qualification matters for layout compatibility.
  static_assert(!__is_layout_compatible(int, _Atomic int));
  static_assert(__is_layout_compatible(_Atomic(int), _Atomic int));
  static_assert(!__is_layout_compatible(int, unsigned int));
  static_assert(!__is_layout_compatible(char, unsigned char));
  static_assert(!__is_layout_compatible(char, signed char));
  static_assert(!__is_layout_compatible(unsigned char, signed char));
  static_assert(__is_layout_compatible(int[], int[]));
  static_assert(__is_layout_compatible(int[2], int[2]));
  static_assert(!__is_layout_compatible(int[n], int[2]));
  // expected-error@-1 {{variable length arrays are not supported in '__is_layout_compatible'}}
  static_assert(!__is_layout_compatible(int[n], int[n]));
  // expected-error@-1 {{variable length arrays are not supported in '__is_layout_compatible'}}
  // expected-error@-2 {{variable length arrays are not supported in '__is_layout_compatible'}}
  static_assert(__is_layout_compatible(int&, int&));
  static_assert(!__is_layout_compatible(int&, char&));
  static_assert(__is_layout_compatible(void(int), void(int)));
  static_assert(!__is_layout_compatible(void(int), void(char)));
  static_assert(__is_layout_compatible(void(&)(int), void(&)(int)));
  static_assert(!__is_layout_compatible(void(&)(int), void(&)(char)));
  static_assert(__is_layout_compatible(void(*)(int), void(*)(int)));
  static_assert(!__is_layout_compatible(void(*)(int), void(*)(char)));
  using function_type = void();
  using function_type2 = void(char);
  static_assert(__is_layout_compatible(const function_type, const function_type));
  // expected-warning@-1 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  // expected-warning@-2 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  static_assert(__is_layout_compatible(function_type, const function_type));
  // expected-warning@-1 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  static_assert(!__is_layout_compatible(const function_type, const function_type2));
  // expected-warning@-1 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  // expected-warning@-2 {{'const' qualifier on function type 'function_type2' (aka 'void (char)') has no effect}}
  static_assert(__is_layout_compatible(CStruct, CStruct2));
  static_assert(__is_layout_compatible(CStruct, const CStruct2));
  static_assert(__is_layout_compatible(CStruct, volatile CStruct2));
  static_assert(__is_layout_compatible(const CStruct, volatile CStruct2));
  static_assert(__is_layout_compatible(CEmptyStruct, CEmptyStruct2));
  static_assert(__is_layout_compatible(CppEmptyStruct, CppEmptyStruct2));
  static_assert(__is_layout_compatible(CppStructStandard, CppStructStandard2));
  static_assert(!__is_layout_compatible(CppStructNonStandardByBase, CppStructNonStandardByBase2));
  static_assert(!__is_layout_compatible(CppStructNonStandardByVirt, CppStructNonStandardByVirt2));
  static_assert(!__is_layout_compatible(CppStructNonStandardByMemb, CppStructNonStandardByMemb2));
  static_assert(!__is_layout_compatible(CppStructNonStandardByProt, CppStructNonStandardByProt2));
  static_assert(!__is_layout_compatible(CppStructNonStandardByVirtBase, CppStructNonStandardByVirtBase2));
  static_assert(!__is_layout_compatible(CppStructNonStandardBySameBase, CppStructNonStandardBySameBase2));
  static_assert(!__is_layout_compatible(CppStructNonStandardBy2ndVirtBase, CppStructNonStandardBy2ndVirtBase2));
  static_assert(__is_layout_compatible(CStruct, CStructWithQualifiers));
  static_assert(__is_layout_compatible(CStruct, CStructNoUniqueAddress) != bool(__has_cpp_attribute(no_unique_address)));
  static_assert(__is_layout_compatible(CStructNoUniqueAddress, CStructNoUniqueAddress2) != bool(__has_cpp_attribute(no_unique_address)));
  static_assert(__is_layout_compatible(CStruct, CStructAlignment));
  static_assert(!__is_layout_compatible(CStruct, CStructAlignedMembers));
  static_assert(__is_layout_compatible(UnionNoOveralignedMembers, UnionWithOveralignedMembers));
  static_assert(__is_layout_compatible(CStructWithBitfelds, CStructWithBitfelds));
  static_assert(__is_layout_compatible(CStructWithBitfelds, CStructWithBitfelds2));
  static_assert(!__is_layout_compatible(CStructWithBitfelds, CStructWithBitfelds3));
  static_assert(!__is_layout_compatible(CStructWithBitfelds, CStructWithBitfelds4));
  static_assert(__is_layout_compatible(int CStruct2::*, int CStruct2::*));
  static_assert(!__is_layout_compatible(int CStruct2::*, char CStruct2::*));
  static_assert(__is_layout_compatible(void(CStruct2::*)(int), void(CStruct2::*)(int)));
  static_assert(!__is_layout_compatible(void(CStruct2::*)(int), void(CStruct2::*)(char)));
  static_assert(__is_layout_compatible(CStructNested, CStructNested2));
  static_assert(__is_layout_compatible(UnionLayout, UnionLayout));
  static_assert(!__is_layout_compatible(UnionLayout, UnionLayout2));
  static_assert(!__is_layout_compatible(UnionLayout, UnionLayout3));
  static_assert(!__is_layout_compatible(StructWithAnonUnion, StructWithAnonUnion2));
  static_assert(!__is_layout_compatible(StructWithAnonUnion, StructWithAnonUnion3));
  static_assert(__is_layout_compatible(EnumLayout, EnumClassLayout));
  static_assert(__is_layout_compatible(EnumForward, EnumForward));
  static_assert(__is_layout_compatible(EnumForward, EnumClassForward));
  static_assert(__is_layout_compatible(CStructIncomplete, CStructIncomplete));
  // expected-error@-1 {{incomplete type 'CStructIncomplete' where a complete type is required}}
  //   expected-note@#CStructIncomplete {{forward declaration of 'CStructIncomplete'}}
  // expected-error@-3 {{incomplete type 'CStructIncomplete' where a complete type is required}}
  //   expected-note@#CStructIncomplete {{forward declaration of 'CStructIncomplete'}}
  static_assert(!__is_layout_compatible(CStruct, CStructIncomplete));
  // expected-error@-1 {{incomplete type 'CStructIncomplete' where a complete type is required}}
  //   expected-note@#CStructIncomplete {{forward declaration of 'CStructIncomplete'}}
  static_assert(__is_layout_compatible(CStructIncomplete[2], CStructIncomplete[2]));
  // expected-error@-1 {{incomplete type 'CStructIncomplete[2]' where a complete type is required}}
  //   expected-note@#CStructIncomplete {{forward declaration of 'CStructIncomplete'}}
  // expected-error@-3 {{incomplete type 'CStructIncomplete[2]' where a complete type is required}}
  //   expected-note@#CStructIncomplete {{forward declaration of 'CStructIncomplete'}}
  static_assert(__is_layout_compatible(CStructIncomplete[], CStructIncomplete[]));
  static_assert(!__is_layout_compatible(CStructWithArrayAtTheEnd, CStructWithFMA));
  static_assert(__is_layout_compatible(CStructWithFMA, CStructWithFMA));
  static_assert(__is_layout_compatible(CStructWithFMA, CStructWithFMA2));
  // Layout compatibility rules for enums might be relaxed in the future. See https://github.com/cplusplus/CWG/issues/39#issuecomment-1184791364
  static_assert(!__is_layout_compatible(EnumLayout, int));
  static_assert(!__is_layout_compatible(EnumClassLayout, int));
  static_assert(!__is_layout_compatible(EnumForward, int));
  static_assert(!__is_layout_compatible(EnumClassForward, int));
  static_assert(__is_layout_compatible(CStruct, D<CStruct>));
  static_assert(__is_layout_compatible(CStruct, D<UniqueEmpty<0>, CStruct>));
  static_assert(__is_layout_compatible(CStruct, D<UniqueEmpty<0>, D<UniqueEmpty<1>, CStruct>, D<UniqueEmpty<2>>>));
  static_assert(__is_layout_compatible(CStruct, D<CStructWithQualifiers>));
  static_assert(__is_layout_compatible(CStruct, D<UniqueEmpty<0>, CStructWithQualifiers>));
  static_assert(__is_layout_compatible(CStructWithQualifiers, D<UniqueEmpty<0>, D<UniqueEmpty<1>, CStruct>, D<UniqueEmpty<2>>>));
}

namespace IPIBO {
struct Base {};
struct Base2 {};
struct Base3 : Base {};
struct Base3Virtual : virtual Base {};
struct Derived : Base {};
struct DerivedIndirect : Base3 {};
struct DerivedMultiple : Base, Base2 {};
struct DerivedAmbiguous : Base, Base3 {};
/* expected-warning@-1 {{direct base 'Base' is inaccessible due to ambiguity:
    struct IPIBO::DerivedAmbiguous -> Base
    struct IPIBO::DerivedAmbiguous -> Base3 -> Base}} */
struct DerivedPrivate : private Base {};
struct DerivedVirtual : virtual Base {};

union Union {};
union UnionIncomplete;
struct StructIncomplete; // #StructIncomplete

void is_pointer_interconvertible_base_of(int n)
{
  static_assert(__is_pointer_interconvertible_base_of(Base, Derived));
  static_assert(!__is_pointer_interconvertible_base_of(Base2, Derived));
  static_assert(__is_pointer_interconvertible_base_of(Base, DerivedIndirect));
  static_assert(__is_pointer_interconvertible_base_of(Base, DerivedMultiple));
  static_assert(__is_pointer_interconvertible_base_of(Base2, DerivedMultiple));
  static_assert(!__is_pointer_interconvertible_base_of(Base3, DerivedMultiple));
  static_assert(!__is_pointer_interconvertible_base_of(Base, DerivedAmbiguous));
  static_assert(__is_pointer_interconvertible_base_of(Base, DerivedPrivate));
  static_assert(!__is_pointer_interconvertible_base_of(Base, DerivedVirtual));
  static_assert(!__is_pointer_interconvertible_base_of(Union, Union));
  static_assert(!__is_pointer_interconvertible_base_of(UnionIncomplete, UnionIncomplete));
  static_assert(__is_pointer_interconvertible_base_of(StructIncomplete, StructIncomplete));
  static_assert(__is_pointer_interconvertible_base_of(StructIncomplete, const StructIncomplete));
  static_assert(__is_pointer_interconvertible_base_of(StructIncomplete, volatile StructIncomplete));
  static_assert(__is_pointer_interconvertible_base_of(const StructIncomplete, volatile StructIncomplete));
  static_assert(!__is_pointer_interconvertible_base_of(StructIncomplete, Derived));
  static_assert(!__is_pointer_interconvertible_base_of(Base, StructIncomplete));
  // expected-error@-1 {{incomplete type 'StructIncomplete' where a complete type is required}}
  //   expected-note@#StructIncomplete {{forward declaration of 'IPIBO::StructIncomplete'}}
  static_assert(!__is_pointer_interconvertible_base_of(CStruct2, CppStructNonStandardByBase2));
  static_assert(!__is_pointer_interconvertible_base_of(void, void));
  static_assert(!__is_pointer_interconvertible_base_of(void, int));
  static_assert(!__is_pointer_interconvertible_base_of(void, const void));
  static_assert(!__is_pointer_interconvertible_base_of(void, volatile void));
  static_assert(!__is_pointer_interconvertible_base_of(const void, volatile void));
  static_assert(!__is_pointer_interconvertible_base_of(int, int));
  static_assert(!__is_pointer_interconvertible_base_of(int, const int));
  static_assert(!__is_pointer_interconvertible_base_of(int, volatile int));
  static_assert(!__is_pointer_interconvertible_base_of(const int, volatile int));
  static_assert(!__is_pointer_interconvertible_base_of(int *, int * __restrict));
  static_assert(!__is_pointer_interconvertible_base_of(int, _Atomic int));
  static_assert(!__is_pointer_interconvertible_base_of(_Atomic(int), _Atomic int));
  static_assert(!__is_pointer_interconvertible_base_of(int, unsigned int));
  static_assert(!__is_pointer_interconvertible_base_of(char, unsigned char));
  static_assert(!__is_pointer_interconvertible_base_of(char, signed char));
  static_assert(!__is_pointer_interconvertible_base_of(unsigned char, signed char));
  using function_type = void();
  using function_type2 = void(char);
  static_assert(!__is_pointer_interconvertible_base_of(const function_type, const function_type));
  // expected-warning@-1 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  // expected-warning@-2 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  static_assert(!__is_pointer_interconvertible_base_of(function_type, const function_type));
  // expected-warning@-1 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  static_assert(!__is_pointer_interconvertible_base_of(const function_type, const function_type2));
  // expected-warning@-1 {{'const' qualifier on function type 'function_type' (aka 'void ()') has no effect}}
  // expected-warning@-2 {{'const' qualifier on function type 'function_type2' (aka 'void (char)') has no effect}}
  static_assert(!__is_pointer_interconvertible_base_of(int CStruct2::*, int CStruct2::*));
  static_assert(!__is_pointer_interconvertible_base_of(int CStruct2::*, char CStruct2::*));
  static_assert(!__is_pointer_interconvertible_base_of(void(CStruct2::*)(int), void(CStruct2::*)(int)));
  static_assert(!__is_pointer_interconvertible_base_of(void(CStruct2::*)(int), void(CStruct2::*)(char)));
  static_assert(!__is_pointer_interconvertible_base_of(int[], int[]));
  static_assert(!__is_pointer_interconvertible_base_of(int[], double[]));
  static_assert(!__is_pointer_interconvertible_base_of(int[2], int[2]));
  static_assert(!__is_pointer_interconvertible_base_of(int[n], int[2]));
  // expected-error@-1 {{variable length arrays are not supported in '__is_pointer_interconvertible_base_of'}}
  static_assert(!__is_pointer_interconvertible_base_of(int[n], int[n]));
  // expected-error@-1 {{variable length arrays are not supported in '__is_pointer_interconvertible_base_of'}}
  // expected-error@-2 {{variable length arrays are not supported in '__is_pointer_interconvertible_base_of'}}
  static_assert(!__is_pointer_interconvertible_base_of(int&, int&));
  static_assert(!__is_pointer_interconvertible_base_of(int&, char&));
  static_assert(!__is_pointer_interconvertible_base_of(void(int), void(int)));
  static_assert(!__is_pointer_interconvertible_base_of(void(int), void(char)));
  static_assert(!__is_pointer_interconvertible_base_of(void(&)(int), void(&)(int)));
  static_assert(!__is_pointer_interconvertible_base_of(void(&)(int), void(&)(char)));
  static_assert(!__is_pointer_interconvertible_base_of(void(*)(int), void(*)(int)));
  static_assert(!__is_pointer_interconvertible_base_of(void(*)(int), void(*)(char)));
}
}

struct NoEligibleTrivialContructor {
  NoEligibleTrivialContructor() {};
  NoEligibleTrivialContructor(const NoEligibleTrivialContructor&) {}
  NoEligibleTrivialContructor(NoEligibleTrivialContructor&&) {}
};

struct OnlyDefaultConstructorIsTrivial {
  OnlyDefaultConstructorIsTrivial() = default;
  OnlyDefaultConstructorIsTrivial(const OnlyDefaultConstructorIsTrivial&) {}
  OnlyDefaultConstructorIsTrivial(OnlyDefaultConstructorIsTrivial&&) {}
};

struct AllContstructorsAreTrivial {
  AllContstructorsAreTrivial() = default;
  AllContstructorsAreTrivial(const AllContstructorsAreTrivial&) = default;
  AllContstructorsAreTrivial(AllContstructorsAreTrivial&&) = default;
};

struct InheritedNoEligibleTrivialConstructor : NoEligibleTrivialContructor {
  using NoEligibleTrivialContructor::NoEligibleTrivialContructor;
};

struct InheritedOnlyDefaultConstructorIsTrivial : OnlyDefaultConstructorIsTrivial {
  using OnlyDefaultConstructorIsTrivial::OnlyDefaultConstructorIsTrivial;
};

struct InheritedAllContstructorsAreTrivial : AllContstructorsAreTrivial {
  using AllContstructorsAreTrivial::AllContstructorsAreTrivial;
};

struct UserDeclaredDestructor {
  ~UserDeclaredDestructor() = default;
};

struct UserProvidedDestructor {
  ~UserProvidedDestructor() {}
};

struct UserDeletedDestructorInAggregate {
  ~UserDeletedDestructorInAggregate() = delete;
};

struct UserDeletedDestructorInNonAggregate {
  virtual void NonAggregate();
  ~UserDeletedDestructorInNonAggregate() = delete;
};

struct DeletedDestructorViaBaseInAggregate : UserDeletedDestructorInAggregate {};
struct DeletedDestructorViaBaseInNonAggregate : UserDeletedDestructorInNonAggregate {};

#if __cplusplus >= 202002L
template<bool B>
struct ConstrainedUserDeclaredDefaultConstructor{
  ConstrainedUserDeclaredDefaultConstructor() requires B = default;
  ConstrainedUserDeclaredDefaultConstructor(const ConstrainedUserDeclaredDefaultConstructor&) {}
};

template<bool B>
struct ConstrainedUserProvidedDestructor {
  ~ConstrainedUserProvidedDestructor() = default;
  ~ConstrainedUserProvidedDestructor() requires B {}
};
#endif

struct StructWithFAM {
  int a[];
};

struct StructWithZeroSizedArray {
  int a[0];
};

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int *align_value_int __attribute__((align_value(16)));

struct [[clang::enforce_read_only_placement]] EnforceReadOnlyPlacement {};
struct [[clang::type_visibility("hidden")]] TypeVisibility {};

void is_implicit_lifetime(int n) {
  static_assert(__builtin_is_implicit_lifetime(decltype(nullptr)));
  static_assert(!__builtin_is_implicit_lifetime(void));
  static_assert(!__builtin_is_implicit_lifetime(const void));
  static_assert(!__builtin_is_implicit_lifetime(volatile void));
  static_assert(__builtin_is_implicit_lifetime(int));
  static_assert(!__builtin_is_implicit_lifetime(int&));
  static_assert(!__builtin_is_implicit_lifetime(int&&));
  static_assert(__builtin_is_implicit_lifetime(float));
  static_assert(__builtin_is_implicit_lifetime(double));
  static_assert(__builtin_is_implicit_lifetime(long double));
  static_assert(__builtin_is_implicit_lifetime(int*));
  static_assert(__builtin_is_implicit_lifetime(int[]));
  static_assert(__builtin_is_implicit_lifetime(int[5]));
  static_assert(__builtin_is_implicit_lifetime(int[n]));
  // expected-error@-1 {{variable length arrays are not supported in '__builtin_is_implicit_lifetime'}}
  static_assert(__builtin_is_implicit_lifetime(Enum));
  static_assert(__builtin_is_implicit_lifetime(EnumClass));
  static_assert(!__builtin_is_implicit_lifetime(void()));
  static_assert(!__builtin_is_implicit_lifetime(void() &));
  static_assert(!__builtin_is_implicit_lifetime(void() const));
  static_assert(!__builtin_is_implicit_lifetime(void(&)()));
  static_assert(__builtin_is_implicit_lifetime(void(*)()));
  static_assert(__builtin_is_implicit_lifetime(decltype(nullptr)));
  static_assert(__builtin_is_implicit_lifetime(int UserDeclaredDestructor::*));
  static_assert(__builtin_is_implicit_lifetime(int (UserDeclaredDestructor::*)()));
  static_assert(__builtin_is_implicit_lifetime(int (UserDeclaredDestructor::*)() const));
  static_assert(__builtin_is_implicit_lifetime(int (UserDeclaredDestructor::*)() &));
  static_assert(__builtin_is_implicit_lifetime(int (UserDeclaredDestructor::*)() &&));
  static_assert(!__builtin_is_implicit_lifetime(IncompleteStruct));
  // expected-error@-1 {{incomplete type 'IncompleteStruct' used in type trait expression}}
  static_assert(__builtin_is_implicit_lifetime(IncompleteStruct[]));
  static_assert(__builtin_is_implicit_lifetime(IncompleteStruct[5]));
  static_assert(__builtin_is_implicit_lifetime(UserDeclaredDestructor));
  static_assert(__builtin_is_implicit_lifetime(const UserDeclaredDestructor));
  static_assert(__builtin_is_implicit_lifetime(volatile UserDeclaredDestructor));
  static_assert(!__builtin_is_implicit_lifetime(UserProvidedDestructor));
  static_assert(!__builtin_is_implicit_lifetime(NoEligibleTrivialContructor));
  static_assert(__builtin_is_implicit_lifetime(OnlyDefaultConstructorIsTrivial));
  static_assert(__builtin_is_implicit_lifetime(AllContstructorsAreTrivial));
  static_assert(!__builtin_is_implicit_lifetime(InheritedNoEligibleTrivialConstructor));
  static_assert(__builtin_is_implicit_lifetime(InheritedOnlyDefaultConstructorIsTrivial));
  static_assert(__builtin_is_implicit_lifetime(InheritedAllContstructorsAreTrivial));
  static_assert(__builtin_is_implicit_lifetime(UserDeletedDestructorInAggregate));
  static_assert(!__builtin_is_implicit_lifetime(UserDeletedDestructorInNonAggregate));
  static_assert(__builtin_is_implicit_lifetime(DeletedDestructorViaBaseInAggregate) == __cplusplus >= 201703L);
  static_assert(!__builtin_is_implicit_lifetime(DeletedDestructorViaBaseInNonAggregate));
#if __cplusplus >= 202002L
  static_assert(__builtin_is_implicit_lifetime(ConstrainedUserDeclaredDefaultConstructor<true>));
  static_assert(!__builtin_is_implicit_lifetime(ConstrainedUserDeclaredDefaultConstructor<false>));
  static_assert(!__builtin_is_implicit_lifetime(ConstrainedUserProvidedDestructor<true>));
  static_assert(__builtin_is_implicit_lifetime(ConstrainedUserProvidedDestructor<false>));
#endif

  static_assert(__builtin_is_implicit_lifetime(__int128));
  static_assert(__builtin_is_implicit_lifetime(_BitInt(8)));
  static_assert(__builtin_is_implicit_lifetime(_BitInt(128)));
  static_assert(__builtin_is_implicit_lifetime(int[0]));
  static_assert(__builtin_is_implicit_lifetime(StructWithFAM));
  static_assert(__builtin_is_implicit_lifetime(StructWithZeroSizedArray));
  static_assert(__builtin_is_implicit_lifetime(__fp16));
  static_assert(__builtin_is_implicit_lifetime(__bf16));
  static_assert(__builtin_is_implicit_lifetime(_Complex double));
  static_assert(__builtin_is_implicit_lifetime(float4));
  static_assert(__builtin_is_implicit_lifetime(align_value_int));
  static_assert(__builtin_is_implicit_lifetime(int[[clang::annotate_type("category2")]] *));
  static_assert(__builtin_is_implicit_lifetime(EnforceReadOnlyPlacement));
  static_assert(__builtin_is_implicit_lifetime(int __attribute__((noderef)) *));
  static_assert(__builtin_is_implicit_lifetime(TypeVisibility));
  static_assert(__builtin_is_implicit_lifetime(int * _Nonnull));
  static_assert(__builtin_is_implicit_lifetime(int * _Null_unspecified));
  static_assert(__builtin_is_implicit_lifetime(int * _Nullable));
  static_assert(!__builtin_is_implicit_lifetime(_Atomic int));
  // expected-error@-1 {{atomic types are not supported in '__builtin_is_implicit_lifetime'}}
  static_assert(__builtin_is_implicit_lifetime(int * __restrict));
}

namespace GH160610 {
class NonAggregate {
public:
    NonAggregate() = default;

    NonAggregate(const NonAggregate&)            = delete;
    NonAggregate& operator=(const NonAggregate&) = delete;
private:
    int num;
};

class DataMemberInitializer {
public:
    DataMemberInitializer() = default;

    DataMemberInitializer(const DataMemberInitializer&)            = delete;
    DataMemberInitializer& operator=(const DataMemberInitializer&) = delete;
private:
    int num = 0;
};

class UserProvidedConstructor {
public:
    UserProvidedConstructor() {}

    UserProvidedConstructor(const UserProvidedConstructor&)            = delete;
    UserProvidedConstructor& operator=(const UserProvidedConstructor&) = delete;
};
struct Ctr {
  Ctr();
};
struct Ctr2 {
  Ctr2();
private:
  NoEligibleTrivialContructor inner;
};

struct NonCopyable{
    NonCopyable() = default;
    NonCopyable(const NonCopyable&) = delete;
};

class C {
    NonCopyable nc;
};

static_assert(__builtin_is_implicit_lifetime(Ctr));
static_assert(!__builtin_is_implicit_lifetime(Ctr2));
static_assert(__builtin_is_implicit_lifetime(C));
static_assert(!__builtin_is_implicit_lifetime(NoEligibleTrivialContructor));
static_assert(__builtin_is_implicit_lifetime(NonAggregate));
static_assert(!__builtin_is_implicit_lifetime(DataMemberInitializer));
static_assert(!__builtin_is_implicit_lifetime(UserProvidedConstructor));

#if __cplusplus >= 202002L
template <typename T>
class Tpl {
    Tpl() requires false = default ;
};
static_assert(__builtin_is_implicit_lifetime(Tpl<int>));

template <typename>
class MultipleDefaults {
  MultipleDefaults() {};
  MultipleDefaults() requires true = default;
};
static_assert(__builtin_is_implicit_lifetime(MultipleDefaults<int>));
template <typename>
class MultipleDefaults2 {
  MultipleDefaults2() requires true {};
  MultipleDefaults2() = default;
};

static_assert(__builtin_is_implicit_lifetime(MultipleDefaults2<int>));


#endif



}

void is_signed()
{
  //static_assert(__is_signed(char));
  static_assert(__is_signed(int));
  static_assert(__is_signed(long));
  static_assert(__is_signed(short));
  static_assert(__is_signed(signed char));
  static_assert(__is_signed(wchar_t));
  static_assert(__is_signed(float));
  static_assert(__is_signed(double));
  static_assert(__is_signed(long double));

  static_assert(!__is_signed(bool));
  static_assert(!__is_signed(cvoid));
  static_assert(!__is_signed(unsigned char));
  static_assert(!__is_signed(unsigned int));
  static_assert(!__is_signed(unsigned long long));
  static_assert(!__is_signed(unsigned long));
  static_assert(!__is_signed(unsigned short));
  static_assert(!__is_signed(void));
  static_assert(!__is_signed(ClassType));
  static_assert(!__is_signed(Derives));
  static_assert(!__is_signed(Enum));
  static_assert(!__is_signed(SignedEnum));
  static_assert(!__is_signed(IntArNB));
  static_assert(!__is_signed(Union));
  static_assert(!__is_signed(UnionAr));
  static_assert(!__is_signed(UnsignedEnum));
}

void is_unsigned()
{
  static_assert(__is_unsigned(bool));
  static_assert(__is_unsigned(unsigned char));
  static_assert(__is_unsigned(unsigned short));
  static_assert(__is_unsigned(unsigned int));
  static_assert(__is_unsigned(unsigned long));
  static_assert(__is_unsigned(unsigned long long));

  static_assert(!__is_unsigned(void));
  static_assert(!__is_unsigned(cvoid));
  static_assert(!__is_unsigned(float));
  static_assert(!__is_unsigned(double));
  static_assert(!__is_unsigned(long double));
  static_assert(!__is_unsigned(char));
  static_assert(!__is_unsigned(signed char));
  static_assert(!__is_unsigned(wchar_t));
  static_assert(!__is_unsigned(short));
  static_assert(!__is_unsigned(int));
  static_assert(!__is_unsigned(long));
  static_assert(!__is_unsigned(Union));
  static_assert(!__is_unsigned(UnionAr));
  static_assert(!__is_unsigned(Derives));
  static_assert(!__is_unsigned(ClassType));
  static_assert(!__is_unsigned(IntArNB));
  static_assert(!__is_unsigned(Enum));
  static_assert(!__is_unsigned(UnsignedEnum));
  static_assert(!__is_unsigned(SignedEnum));
}

typedef Int& IntRef;
typedef const IntAr ConstIntAr;
typedef ConstIntAr ConstIntArAr[4];

struct HasCopy {
  HasCopy(HasCopy& cp);
};

struct HasMove {
  HasMove(HasMove&& cp);
};

struct HasTemplateCons {
  HasVirt Annoying;

  template <typename T>
  HasTemplateCons(const T&);
};

void has_trivial_default_constructor() {
  static_assert(__has_trivial_constructor(Int));
  static_assert(__has_trivial_constructor(IntAr));
  static_assert(__has_trivial_constructor(Union));
  static_assert(__has_trivial_constructor(UnionAr));
  static_assert(__has_trivial_constructor(POD));
  static_assert(__has_trivial_constructor(Derives));
  static_assert(__has_trivial_constructor(DerivesAr));
  static_assert(__has_trivial_constructor(ConstIntAr));
  static_assert(__has_trivial_constructor(ConstIntArAr));
  static_assert(__has_trivial_constructor(HasDest));
  static_assert(__has_trivial_constructor(HasPriv));
  static_assert(__has_trivial_constructor(HasCopyAssign));
  static_assert(__has_trivial_constructor(HasMoveAssign));
  static_assert(__has_trivial_constructor(const Int));
  static_assert(__has_trivial_constructor(AllDefaulted));
  static_assert(__has_trivial_constructor(AllDeleted));
  static_assert(__has_trivial_constructor(ACompleteType[]));

  static_assert(!__has_trivial_constructor(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_trivial_constructor(HasCons));
  static_assert(!__has_trivial_constructor(HasRef));
  static_assert(!__has_trivial_constructor(HasCopy));
  static_assert(!__has_trivial_constructor(IntRef));
  static_assert(!__has_trivial_constructor(VirtAr));
  static_assert(!__has_trivial_constructor(void));
  static_assert(!__has_trivial_constructor(cvoid));
  static_assert(!__has_trivial_constructor(HasTemplateCons));
  static_assert(!__has_trivial_constructor(AllPrivate));
  static_assert(!__has_trivial_constructor(ExtDefaulted));
}

void has_trivial_move_constructor() {
  // n3376 12.8 [class.copy]/12
  // A copy/move constructor for class X is trivial if it is not
  // user-provided, its declared parameter type is the same as
  // if it had been implicitly declared, and if
  //   - class X has no virtual functions (10.3) and no virtual
  //     base classes (10.1), and
  //   - the constructor selected to copy/move each direct base
  //     class subobject is trivial, and
  //   - for each non-static data member of X that is of class
  //     type (or array thereof), the constructor selected
  //     to copy/move that member is trivial;
  // otherwise the copy/move constructor is non-trivial.
  static_assert(__has_trivial_move_constructor(POD));
  static_assert(__has_trivial_move_constructor(Union));
  static_assert(__has_trivial_move_constructor(HasCons));
  static_assert(__has_trivial_move_constructor(HasStaticMemberMoveCtor));
  static_assert(__has_trivial_move_constructor(AllDeleted));
  static_assert(__has_trivial_move_constructor(ACompleteType[]));

  static_assert(!__has_trivial_move_constructor(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_trivial_move_constructor(HasVirt));
  static_assert(!__has_trivial_move_constructor(DerivesVirt));
  static_assert(!__has_trivial_move_constructor(HasMoveCtor));
  static_assert(!__has_trivial_move_constructor(DerivesHasMoveCtor));
  static_assert(!__has_trivial_move_constructor(HasMemberMoveCtor));
}

void has_trivial_copy_constructor() {
  static_assert(__has_trivial_copy(Int));
  static_assert(__has_trivial_copy(IntAr));
  static_assert(__has_trivial_copy(Union));
  static_assert(__has_trivial_copy(UnionAr));
  static_assert(__has_trivial_copy(POD));
  static_assert(__has_trivial_copy(Derives));
  static_assert(__has_trivial_copy(ConstIntAr));
  static_assert(__has_trivial_copy(ConstIntArAr));
  static_assert(__has_trivial_copy(HasDest));
  static_assert(__has_trivial_copy(HasPriv));
  static_assert(__has_trivial_copy(HasCons));
  static_assert(__has_trivial_copy(HasRef));
  static_assert(__has_trivial_copy(HasMove));
  static_assert(__has_trivial_copy(IntRef));
  static_assert(__has_trivial_copy(HasCopyAssign));
  static_assert(__has_trivial_copy(HasMoveAssign));
  static_assert(__has_trivial_copy(const Int));
  static_assert(__has_trivial_copy(AllDefaulted));
  static_assert(__has_trivial_copy(AllDeleted));
  static_assert(__has_trivial_copy(DerivesAr));
  static_assert(__has_trivial_copy(DerivesHasRef));
  static_assert(__has_trivial_copy(ACompleteType[]));

  static_assert(!__has_trivial_copy(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_trivial_copy(HasCopy));
  static_assert(!__has_trivial_copy(HasTemplateCons));
  static_assert(!__has_trivial_copy(VirtAr));
  static_assert(!__has_trivial_copy(void));
  static_assert(!__has_trivial_copy(cvoid));
  static_assert(!__has_trivial_copy(AllPrivate));
  static_assert(!__has_trivial_copy(ExtDefaulted));
}

void has_trivial_copy_assignment() {
  static_assert(__has_trivial_assign(Int));
  static_assert(__has_trivial_assign(IntAr));
  static_assert(__has_trivial_assign(Union));
  static_assert(__has_trivial_assign(UnionAr));
  static_assert(__has_trivial_assign(POD));
  static_assert(__has_trivial_assign(Derives));
  static_assert(__has_trivial_assign(HasDest));
  static_assert(__has_trivial_assign(HasPriv));
  static_assert(__has_trivial_assign(HasCons));
  static_assert(__has_trivial_assign(HasRef));
  static_assert(__has_trivial_assign(HasCopy));
  static_assert(__has_trivial_assign(HasMove));
  static_assert(__has_trivial_assign(HasMoveAssign));
  static_assert(__has_trivial_assign(AllDefaulted));
  static_assert(__has_trivial_assign(AllDeleted));
  static_assert(__has_trivial_assign(DerivesAr));
  static_assert(__has_trivial_assign(DerivesHasRef));
  static_assert(__has_trivial_assign(ACompleteType[]));

  static_assert(!__has_trivial_assign(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_trivial_assign(IntRef));
  static_assert(!__has_trivial_assign(HasCopyAssign));
  static_assert(!__has_trivial_assign(const Int));
  static_assert(!__has_trivial_assign(ConstIntAr));
  static_assert(!__has_trivial_assign(ConstIntArAr));
  static_assert(!__has_trivial_assign(VirtAr));
  static_assert(!__has_trivial_assign(void));
  static_assert(!__has_trivial_assign(cvoid));
  static_assert(!__has_trivial_assign(AllPrivate));
  static_assert(!__has_trivial_assign(ExtDefaulted));
}

void has_trivial_destructor() {
  static_assert(__has_trivial_destructor(Int));
  static_assert(__has_trivial_destructor(IntAr));
  static_assert(__has_trivial_destructor(Union));
  static_assert(__has_trivial_destructor(UnionAr));
  static_assert(__has_trivial_destructor(POD));
  static_assert(__has_trivial_destructor(Derives));
  static_assert(__has_trivial_destructor(ConstIntAr));
  static_assert(__has_trivial_destructor(ConstIntArAr));
  static_assert(__has_trivial_destructor(HasPriv));
  static_assert(__has_trivial_destructor(HasCons));
  static_assert(__has_trivial_destructor(HasRef));
  static_assert(__has_trivial_destructor(HasCopy));
  static_assert(__has_trivial_destructor(HasMove));
  static_assert(__has_trivial_destructor(IntRef));
  static_assert(__has_trivial_destructor(HasCopyAssign));
  static_assert(__has_trivial_destructor(HasMoveAssign));
  static_assert(__has_trivial_destructor(const Int));
  static_assert(__has_trivial_destructor(DerivesAr));
  static_assert(__has_trivial_destructor(VirtAr));
  static_assert(__has_trivial_destructor(AllDefaulted));
  static_assert(__has_trivial_destructor(AllDeleted));
  static_assert(__has_trivial_destructor(DerivesHasRef));
  static_assert(__has_trivial_destructor(ACompleteType[]));

  static_assert(!__has_trivial_destructor(HasDest));
  static_assert(!__has_trivial_destructor(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_trivial_destructor(void));
  static_assert(!__has_trivial_destructor(cvoid));
  static_assert(!__has_trivial_destructor(AllPrivate));
  static_assert(!__has_trivial_destructor(ExtDefaulted));
}

struct A { ~A() {} };
template<typename> struct B : A { };

void f() {
  static_assert(!__has_trivial_destructor(A));
  static_assert(!__has_trivial_destructor(B<int>));
}

class PR11110 {
  template <int> int operator=( int );
  int operator=(PR11110);
};

class UsingAssign;

class UsingAssignBase {
protected:
  UsingAssign &operator=(const UsingAssign&) throw();
};

class UsingAssign : public UsingAssignBase {
public:
  using UsingAssignBase::operator=;
};

void has_nothrow_assign() {
  static_assert(__has_nothrow_assign(Int));
  static_assert(__has_nothrow_assign(IntAr));
  static_assert(__has_nothrow_assign(Union));
  static_assert(__has_nothrow_assign(UnionAr));
  static_assert(__has_nothrow_assign(POD));
  static_assert(__has_nothrow_assign(Derives));
  static_assert(__has_nothrow_assign(HasDest));
  static_assert(__has_nothrow_assign(HasPriv));
  static_assert(__has_nothrow_assign(HasCons));
  static_assert(__has_nothrow_assign(HasRef));
  static_assert(__has_nothrow_assign(HasCopy));
  static_assert(__has_nothrow_assign(HasMove));
  static_assert(__has_nothrow_assign(HasMoveAssign));
  static_assert(__has_nothrow_assign(HasNoThrowCopyAssign));
  static_assert(__has_nothrow_assign(HasMultipleNoThrowCopyAssign));
  static_assert(__has_nothrow_assign(HasVirtDest));
  static_assert(__has_nothrow_assign(AllPrivate));
  static_assert(__has_nothrow_assign(UsingAssign));
  static_assert(__has_nothrow_assign(DerivesAr));
  static_assert(__has_nothrow_assign(ACompleteType[]));

  static_assert(!__has_nothrow_assign(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_nothrow_assign(IntRef));
  static_assert(!__has_nothrow_assign(HasCopyAssign));
  static_assert(!__has_nothrow_assign(HasMultipleCopyAssign));
  static_assert(!__has_nothrow_assign(const Int));
  static_assert(!__has_nothrow_assign(ConstIntAr));
  static_assert(!__has_nothrow_assign(ConstIntArAr));
  static_assert(!__has_nothrow_assign(VirtAr));
  static_assert(!__has_nothrow_assign(void));
  static_assert(!__has_nothrow_assign(cvoid));
  static_assert(!__has_nothrow_assign(PR11110));
}

void has_nothrow_move_assign() {
  static_assert(__has_nothrow_move_assign(Int));
  static_assert(__has_nothrow_move_assign(Enum));
  static_assert(__has_nothrow_move_assign(Int*));
  static_assert(__has_nothrow_move_assign(Enum POD::*));
  static_assert(__has_nothrow_move_assign(POD));
  static_assert(__has_nothrow_move_assign(HasPriv));
  static_assert(__has_nothrow_move_assign(HasNoThrowMoveAssign));
  static_assert(__has_nothrow_move_assign(HasNoExceptNoThrowMoveAssign));
  static_assert(__has_nothrow_move_assign(HasMemberNoThrowMoveAssign));
  static_assert(__has_nothrow_move_assign(HasMemberNoExceptNoThrowMoveAssign));
  static_assert(__has_nothrow_move_assign(AllDeleted));
  static_assert(__has_nothrow_move_assign(ACompleteType[]));

  static_assert(!__has_nothrow_move_assign(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_nothrow_move_assign(HasThrowMoveAssign));
  static_assert(!__has_nothrow_move_assign(HasNoExceptFalseMoveAssign));
  static_assert(!__has_nothrow_move_assign(HasMemberThrowMoveAssign));
  static_assert(!__has_nothrow_move_assign(HasMemberNoExceptFalseMoveAssign));
  static_assert(!__has_nothrow_move_assign(NoDefaultMoveAssignDueToUDCopyCtor));
  static_assert(!__has_nothrow_move_assign(NoDefaultMoveAssignDueToUDCopyAssign));
  static_assert(!__has_nothrow_move_assign(NoDefaultMoveAssignDueToDtor));


  static_assert(__is_nothrow_assignable(HasNoThrowMoveAssign, HasNoThrowMoveAssign));
  static_assert(!__is_nothrow_assignable(HasThrowMoveAssign, HasThrowMoveAssign));

  static_assert(__is_assignable(HasNoThrowMoveAssign, HasNoThrowMoveAssign));
  static_assert(__is_assignable(HasThrowMoveAssign, HasThrowMoveAssign));
}

void has_trivial_move_assign() {
  // n3376 12.8 [class.copy]/25
  // A copy/move assignment operator for class X is trivial if it
  // is not user-provided, its declared parameter type is the same
  // as if it had been implicitly declared, and if:
  //  - class X has no virtual functions (10.3) and no virtual base
  //    classes (10.1), and
  //  - the assignment operator selected to copy/move each direct
  //    base class subobject is trivial, and
  //  - for each non-static data member of X that is of class type
  //    (or array thereof), the assignment operator
  //    selected to copy/move that member is trivial;
  static_assert(__has_trivial_move_assign(Int));
  static_assert(__has_trivial_move_assign(HasStaticMemberMoveAssign));
  static_assert(__has_trivial_move_assign(AllDeleted));
  static_assert(__has_trivial_move_assign(ACompleteType[]));

  static_assert(!__has_trivial_move_assign(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_trivial_move_assign(HasVirt));
  static_assert(!__has_trivial_move_assign(DerivesVirt));
  static_assert(!__has_trivial_move_assign(HasMoveAssign));
  static_assert(!__has_trivial_move_assign(DerivesHasMoveAssign));
  static_assert(!__has_trivial_move_assign(HasMemberMoveAssign));
  static_assert(!__has_nothrow_move_assign(NoDefaultMoveAssignDueToUDCopyCtor));
  static_assert(!__has_nothrow_move_assign(NoDefaultMoveAssignDueToUDCopyAssign));
}

void has_nothrow_copy() {
  static_assert(__has_nothrow_copy(Int));
  static_assert(__has_nothrow_copy(IntAr));
  static_assert(__has_nothrow_copy(Union));
  static_assert(__has_nothrow_copy(UnionAr));
  static_assert(__has_nothrow_copy(POD));
  static_assert(__has_nothrow_copy(const Int));
  static_assert(__has_nothrow_copy(ConstIntAr));
  static_assert(__has_nothrow_copy(ConstIntArAr));
  static_assert(__has_nothrow_copy(Derives));
  static_assert(__has_nothrow_copy(IntRef));
  static_assert(__has_nothrow_copy(HasDest));
  static_assert(__has_nothrow_copy(HasPriv));
  static_assert(__has_nothrow_copy(HasCons));
  static_assert(__has_nothrow_copy(HasRef));
  static_assert(__has_nothrow_copy(HasMove));
  static_assert(__has_nothrow_copy(HasCopyAssign));
  static_assert(__has_nothrow_copy(HasMoveAssign));
  static_assert(__has_nothrow_copy(HasNoThrowCopy));
  static_assert(__has_nothrow_copy(HasMultipleNoThrowCopy));
  static_assert(__has_nothrow_copy(HasVirtDest));
  static_assert(__has_nothrow_copy(HasTemplateCons));
  static_assert(__has_nothrow_copy(AllPrivate));
  static_assert(__has_nothrow_copy(DerivesAr));
  static_assert(__has_nothrow_copy(ACompleteType[]));

  static_assert(!__has_nothrow_copy(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_nothrow_copy(HasCopy));
  static_assert(!__has_nothrow_copy(HasMultipleCopy));
  static_assert(!__has_nothrow_copy(VirtAr));
  static_assert(!__has_nothrow_copy(void));
  static_assert(!__has_nothrow_copy(cvoid));
}

void has_nothrow_constructor() {
  static_assert(__has_nothrow_constructor(Int));
  static_assert(__has_nothrow_constructor(IntAr));
  static_assert(__has_nothrow_constructor(Union));
  static_assert(__has_nothrow_constructor(UnionAr));
  static_assert(__has_nothrow_constructor(POD));
  static_assert(__has_nothrow_constructor(Derives));
  static_assert(__has_nothrow_constructor(DerivesAr));
  static_assert(__has_nothrow_constructor(ConstIntAr));
  static_assert(__has_nothrow_constructor(ConstIntArAr));
  static_assert(__has_nothrow_constructor(HasDest));
  static_assert(__has_nothrow_constructor(HasPriv));
  static_assert(__has_nothrow_constructor(HasCopyAssign));
  static_assert(__has_nothrow_constructor(const Int));
  static_assert(__has_nothrow_constructor(HasNoThrowConstructor));
  static_assert(__has_nothrow_constructor(HasVirtDest));
  // static_assert(__has_nothrow_constructor(VirtAr)); // not implemented
  static_assert(__has_nothrow_constructor(AllPrivate));
  static_assert(__has_nothrow_constructor(ACompleteType[]));

  static_assert(!__has_nothrow_constructor(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(!__has_nothrow_constructor(HasCons));
  static_assert(!__has_nothrow_constructor(HasRef));
  static_assert(!__has_nothrow_constructor(HasCopy));
  static_assert(!__has_nothrow_constructor(HasMove));
  static_assert(!__has_nothrow_constructor(HasNoThrowConstructorWithArgs));
  static_assert(!__has_nothrow_constructor(IntRef));
  static_assert(!__has_nothrow_constructor(void));
  static_assert(!__has_nothrow_constructor(cvoid));
  static_assert(!__has_nothrow_constructor(HasTemplateCons));

  static_assert(!__has_nothrow_constructor(HasMultipleDefaultConstructor1));
  static_assert(!__has_nothrow_constructor(HasMultipleDefaultConstructor2));
}

void has_virtual_destructor() {
  static_assert(!__has_virtual_destructor(Int));
  static_assert(!__has_virtual_destructor(IntAr));
  static_assert(!__has_virtual_destructor(Union));
  static_assert(!__has_virtual_destructor(UnionAr));
  static_assert(!__has_virtual_destructor(POD));
  static_assert(!__has_virtual_destructor(Derives));
  static_assert(!__has_virtual_destructor(DerivesAr));
  static_assert(!__has_virtual_destructor(const Int));
  static_assert(!__has_virtual_destructor(ConstIntAr));
  static_assert(!__has_virtual_destructor(ConstIntArAr));
  static_assert(!__has_virtual_destructor(HasDest));
  static_assert(!__has_virtual_destructor(HasPriv));
  static_assert(!__has_virtual_destructor(HasCons));
  static_assert(!__has_virtual_destructor(HasRef));
  static_assert(!__has_virtual_destructor(HasCopy));
  static_assert(!__has_virtual_destructor(HasMove));
  static_assert(!__has_virtual_destructor(HasCopyAssign));
  static_assert(!__has_virtual_destructor(HasMoveAssign));
  static_assert(!__has_virtual_destructor(IntRef));
  static_assert(!__has_virtual_destructor(VirtAr));
  static_assert(!__has_virtual_destructor(ACompleteType[]));

  static_assert(!__has_virtual_destructor(AnIncompleteType[])); // expected-error {{incomplete type}}
  static_assert(__has_virtual_destructor(HasVirtDest));
  static_assert(__has_virtual_destructor(DerivedVirtDest));
  static_assert(!__has_virtual_destructor(VirtDestAr));
  static_assert(!__has_virtual_destructor(void));
  static_assert(!__has_virtual_destructor(cvoid));
  static_assert(!__has_virtual_destructor(AllPrivate));
}


class Base {};
class Derived : Base {};
class Derived2a : Derived {};
class Derived2b : Derived {};
class Derived3 : virtual Derived2a, virtual Derived2b {};
template<typename T> struct BaseA { T a;  };
template<typename T> struct DerivedB : BaseA<T> { };
template<typename T> struct CrazyDerived : T { };


class class_forward; // expected-note 4 {{forward declaration of 'class_forward'}}

template <class T> class DerivedTemp : Base {};
template <class T> class NonderivedTemp {};
template <class T> class UndefinedTemp; // expected-note 2 {{declared here}}

void is_base_of() {
  static_assert(__is_base_of(Base, Derived));
  static_assert(__is_base_of(const Base, Derived));
  static_assert(!__is_base_of(Derived, Base));
  static_assert(!__is_base_of(Derived, int));
  static_assert(__is_base_of(Base, Base));
  static_assert(__is_base_of(Base, Derived3));
  static_assert(__is_base_of(Derived, Derived3));
  static_assert(__is_base_of(Derived2b, Derived3));
  static_assert(__is_base_of(Derived2a, Derived3));
  static_assert(__is_base_of(BaseA<int>, DerivedB<int>));
  static_assert(!__is_base_of(DerivedB<int>, BaseA<int>));
  static_assert(__is_base_of(Base, CrazyDerived<Base>));
  static_assert(!__is_base_of(Union, Union));
  static_assert(__is_base_of(Empty, Empty));
  static_assert(__is_base_of(class_forward, class_forward));
  static_assert(!__is_base_of(Empty, class_forward)); // expected-error {{incomplete type 'class_forward' used in type trait expression}}
  static_assert(!__is_base_of(Base&, Derived&));
  static_assert(!__is_base_of(Base[10], Derived[10]));
  static_assert(!__is_base_of(int, int));
  static_assert(!__is_base_of(long, int));
  static_assert(__is_base_of(Base, DerivedTemp<int>));
  static_assert(!__is_base_of(Base, NonderivedTemp<int>));
  static_assert(!__is_base_of(Base, UndefinedTemp<int>)); // expected-error {{implicit instantiation of undefined template 'UndefinedTemp<int>'}}

  static_assert(!__is_base_of(IncompleteUnion, IncompleteUnion));
  static_assert(!__is_base_of(Union, IncompleteUnion));
  static_assert(!__is_base_of(IncompleteUnion, Union));
  static_assert(!__is_base_of(IncompleteStruct, IncompleteUnion));
  static_assert(!__is_base_of(IncompleteUnion, IncompleteStruct));
  static_assert(!__is_base_of(Empty, IncompleteUnion));
  static_assert(!__is_base_of(IncompleteUnion, Empty));
  static_assert(!__is_base_of(int, IncompleteUnion));
  static_assert(!__is_base_of(IncompleteUnion, int));
  static_assert(!__is_base_of(Empty, Union));
  static_assert(!__is_base_of(Union, Empty));
  static_assert(!__is_base_of(int, Empty));
  static_assert(!__is_base_of(Union, int));

  static_assert(__is_base_of(Base, Derived));
  static_assert(!__is_base_of(Derived, Base));

  static_assert(__is_base_of(Base, CrazyDerived<Base>));
  static_assert(!__is_base_of(CrazyDerived<Base>, Base));

  static_assert(__is_base_of(BaseA<int>, DerivedB<int>));
  static_assert(!__is_base_of(DerivedB<int>, BaseA<int>));
}

struct DerivedTransitiveViaNonVirtual : Derived3 {};
struct DerivedTransitiveViaVirtual : virtual Derived3 {};

template <typename T>
struct CrazyDerivedVirtual : virtual T {};

struct DerivedPrivate : private virtual Base {};
struct DerivedProtected : protected virtual Base {};
struct DerivedPrivatePrivate : private DerivedPrivate {};
struct DerivedPrivateProtected : private DerivedProtected {};
struct DerivedProtectedPrivate : protected DerivedProtected {};
struct DerivedProtectedProtected : protected DerivedProtected {};

void is_virtual_base_of(int n) {
  static_assert(!__builtin_is_virtual_base_of(Base, Derived));
  static_assert(!__builtin_is_virtual_base_of(const Base, Derived));
  static_assert(!__builtin_is_virtual_base_of(Derived, Base));
  static_assert(!__builtin_is_virtual_base_of(Derived, int));
  static_assert(!__builtin_is_virtual_base_of(Base, Base));
  static_assert(!__builtin_is_virtual_base_of(Base, Derived3));
  static_assert(!__builtin_is_virtual_base_of(Derived, Derived3));
  static_assert(__builtin_is_virtual_base_of(Derived2b, Derived3));
  static_assert(__builtin_is_virtual_base_of(Derived2a, Derived3));
  static_assert(!__builtin_is_virtual_base_of(BaseA<int>, DerivedB<int>));
  static_assert(!__builtin_is_virtual_base_of(DerivedB<int>, BaseA<int>));
  static_assert(!__builtin_is_virtual_base_of(Union, Union));
  static_assert(!__builtin_is_virtual_base_of(Empty, Empty));
  static_assert(!__builtin_is_virtual_base_of(class_forward, class_forward)); // expected-error {{incomplete type 'class_forward' where a complete type is required}}
  static_assert(!__builtin_is_virtual_base_of(Empty, class_forward)); // expected-error {{incomplete type 'class_forward' where a complete type is required}}
  static_assert(!__builtin_is_virtual_base_of(class_forward, Empty));
  static_assert(!__builtin_is_virtual_base_of(Base&, Derived&));
  static_assert(!__builtin_is_virtual_base_of(Base[10], Derived[10]));
  static_assert(!__builtin_is_virtual_base_of(Base[n], Derived[n])); // expected-error 2 {{variable length arrays are not supported in '__builtin_is_virtual_base_of'}}
  static_assert(!__builtin_is_virtual_base_of(int, int));
  static_assert(!__builtin_is_virtual_base_of(int[], int[]));
  static_assert(!__builtin_is_virtual_base_of(long, int));
  static_assert(!__builtin_is_virtual_base_of(Base, DerivedTemp<int>));
  static_assert(!__builtin_is_virtual_base_of(Base, NonderivedTemp<int>));
  static_assert(!__builtin_is_virtual_base_of(Base, UndefinedTemp<int>)); // expected-error {{implicit instantiation of undefined template 'UndefinedTemp<int>'}}
  static_assert(__builtin_is_virtual_base_of(Base, DerivedPrivate));
  static_assert(__builtin_is_virtual_base_of(Base, DerivedProtected));
  static_assert(__builtin_is_virtual_base_of(Base, DerivedPrivatePrivate));
  static_assert(__builtin_is_virtual_base_of(Base, DerivedPrivateProtected));
  static_assert(__builtin_is_virtual_base_of(Base, DerivedProtectedPrivate));
  static_assert(__builtin_is_virtual_base_of(Base, DerivedProtectedProtected));
  static_assert(__builtin_is_virtual_base_of(Derived2a, DerivedTransitiveViaNonVirtual));
  static_assert(__builtin_is_virtual_base_of(Derived2b, DerivedTransitiveViaNonVirtual));
  static_assert(__builtin_is_virtual_base_of(Derived2a, DerivedTransitiveViaVirtual));
  static_assert(__builtin_is_virtual_base_of(Derived2b, DerivedTransitiveViaVirtual));
  static_assert(!__builtin_is_virtual_base_of(Base, CrazyDerived<Base>));
  static_assert(!__builtin_is_virtual_base_of(CrazyDerived<Base>, Base));
  static_assert(__builtin_is_virtual_base_of(Base, CrazyDerivedVirtual<Base>));
  static_assert(!__builtin_is_virtual_base_of(CrazyDerivedVirtual<Base>, Base));

  static_assert(!__builtin_is_virtual_base_of(IncompleteUnion, IncompleteUnion));
  static_assert(!__builtin_is_virtual_base_of(Union, IncompleteUnion));
  static_assert(!__builtin_is_virtual_base_of(IncompleteUnion, Union));
  static_assert(!__builtin_is_virtual_base_of(IncompleteStruct, IncompleteUnion));
  static_assert(!__builtin_is_virtual_base_of(IncompleteUnion, IncompleteStruct));
  static_assert(!__builtin_is_virtual_base_of(Empty, IncompleteUnion));
  static_assert(!__builtin_is_virtual_base_of(IncompleteUnion, Empty));
  static_assert(!__builtin_is_virtual_base_of(int, IncompleteUnion));
  static_assert(!__builtin_is_virtual_base_of(IncompleteUnion, int));
  static_assert(!__builtin_is_virtual_base_of(Empty, Union));
  static_assert(!__builtin_is_virtual_base_of(Union, Empty));
  static_assert(!__builtin_is_virtual_base_of(int, Empty));
  static_assert(!__builtin_is_virtual_base_of(Union, int));
  static_assert(!__builtin_is_virtual_base_of(IncompleteStruct, IncompleteStruct[n])); // expected-error {{variable length arrays are not supported in '__builtin_is_virtual_base_of'}}
}

template<class T, class U>
class TemplateClass {};

template<class T>
using TemplateAlias = TemplateClass<T, int>;

typedef class Base BaseTypedef;

void is_same()
{
  static_assert(__is_same(Base, Base));
  static_assert(__is_same(Base, BaseTypedef));
  static_assert(__is_same(TemplateClass<int, int>, TemplateAlias<int>));

  static_assert(!__is_same(Base, const Base));
  static_assert(!__is_same(Base, Base&));
  static_assert(!__is_same(Base, Derived));

  // __is_same_as is a GCC compatibility synonym for __is_same.
  static_assert(__is_same_as(int, int));
  static_assert(!__is_same_as(int, float));
}

struct IntWrapper
{
  int value;
  IntWrapper(int _value) : value(_value) {}
  operator int() const noexcept {
    return value;
  }
};

struct FloatWrapper
{
  float value;
  FloatWrapper(float _value) noexcept : value(_value) {}
  FloatWrapper(const IntWrapper& obj)
    : value(static_cast<float>(obj.value)) {}
  operator float() const {
    return value;
  }
  operator IntWrapper() const {
    return IntWrapper(static_cast<int>(value));
  }
};

template<typename A, typename B, bool result = __is_convertible(A, B)>
static constexpr bool is_convertible_sfinae() { return result; }

void is_convertible()
{
  static_assert(__is_convertible(IntWrapper, IntWrapper));
  static_assert(__is_convertible(IntWrapper, const IntWrapper));
  static_assert(__is_convertible(IntWrapper, int));
  static_assert(__is_convertible(int, IntWrapper));
  static_assert(__is_convertible(IntWrapper, FloatWrapper));
  static_assert(__is_convertible(FloatWrapper, IntWrapper));
  static_assert(__is_convertible(FloatWrapper, float));
  static_assert(__is_convertible(float, FloatWrapper));
  static_assert(__is_convertible(IntWrapper, IntWrapper&&));
  static_assert(__is_convertible(IntWrapper, const IntWrapper&));
  static_assert(__is_convertible(IntWrapper, int&&));
  static_assert(__is_convertible(IntWrapper, const int&));
  static_assert(__is_convertible(int, IntWrapper&&));
  static_assert(__is_convertible(int, const IntWrapper&));
  static_assert(__is_convertible(IntWrapper, FloatWrapper&&));
  static_assert(__is_convertible(IntWrapper, const FloatWrapper&));
  static_assert(__is_convertible(FloatWrapper, IntWrapper&&));
  static_assert(__is_convertible(FloatWrapper, const IntWrapper&&));
  static_assert(__is_convertible(FloatWrapper, float&&));
  static_assert(__is_convertible(FloatWrapper, const float&));
  static_assert(__is_convertible(float, FloatWrapper&&));
  static_assert(__is_convertible(float, const FloatWrapper&));

  static_assert(!__is_convertible(AllPrivate, AllPrivate));
  // Make sure we don't emit "calling a private constructor" in SFINAE context.
  static_assert(!is_convertible_sfinae<AllPrivate, AllPrivate>());
}

void is_nothrow_convertible()
{
  static_assert(__is_nothrow_convertible(IntWrapper, IntWrapper));
  static_assert(__is_nothrow_convertible(IntWrapper, const IntWrapper));
  static_assert(__is_nothrow_convertible(IntWrapper, int));
  static_assert(!__is_nothrow_convertible(int, IntWrapper));
  static_assert(!__is_nothrow_convertible(IntWrapper, FloatWrapper));
  static_assert(!__is_nothrow_convertible(FloatWrapper, IntWrapper));
  static_assert(!__is_nothrow_convertible(FloatWrapper, float));
  static_assert(__is_nothrow_convertible(float, FloatWrapper));
  static_assert(__is_nothrow_convertible(IntWrapper, IntWrapper&&));
  static_assert(__is_nothrow_convertible(IntWrapper, const IntWrapper&));
  static_assert(__is_nothrow_convertible(IntWrapper, int&&));
  static_assert(__is_nothrow_convertible(IntWrapper, const int&));
  static_assert(!__is_nothrow_convertible(int, IntWrapper&&));
  static_assert(!__is_nothrow_convertible(int, const IntWrapper&));
  static_assert(!__is_nothrow_convertible(IntWrapper, FloatWrapper&&));
  static_assert(!__is_nothrow_convertible(IntWrapper, const FloatWrapper&));
  static_assert(!__is_nothrow_convertible(FloatWrapper, IntWrapper&&));
  static_assert(!__is_nothrow_convertible(FloatWrapper, const IntWrapper&));
  static_assert(!__is_nothrow_convertible(FloatWrapper, float&&));
  static_assert(!__is_nothrow_convertible(FloatWrapper, const float&));
  static_assert(__is_nothrow_convertible(float, FloatWrapper&&));
  static_assert(__is_nothrow_convertible(float, const FloatWrapper&));
}

struct FromInt { FromInt(int); };
struct ToInt { operator int(); };
typedef void Function();

void is_convertible_to();
class PrivateCopy {
  PrivateCopy(const PrivateCopy&);
  friend void is_convertible_to();
};

template<typename T>
struct X0 {
  template<typename U> X0(const X0<U>&);
};

struct Abstract { virtual void f() = 0; };

void is_convertible_to() {
  static_assert(__is_convertible_to(Int, Int));
  static_assert(!__is_convertible_to(Int, IntAr));
  static_assert(!__is_convertible_to(IntAr, IntAr));
  static_assert(__is_convertible_to(void, void));
  static_assert(__is_convertible_to(cvoid, void));
  static_assert(__is_convertible_to(void, cvoid));
  static_assert(__is_convertible_to(cvoid, cvoid));
  static_assert(__is_convertible_to(int, FromInt));
  static_assert(__is_convertible_to(long, FromInt));
  static_assert(__is_convertible_to(double, FromInt));
  static_assert(__is_convertible_to(const int, FromInt));
  static_assert(__is_convertible_to(const int&, FromInt));
  static_assert(__is_convertible_to(ToInt, int));
  static_assert(__is_convertible_to(ToInt, const int&));
  static_assert(__is_convertible_to(ToInt, long));
  static_assert(!__is_convertible_to(ToInt, int&));
  static_assert(!__is_convertible_to(ToInt, FromInt));
  static_assert(__is_convertible_to(IntAr&, IntAr&));
  static_assert(__is_convertible_to(IntAr&, const IntAr&));
  static_assert(!__is_convertible_to(const IntAr&, IntAr&));
  static_assert(!__is_convertible_to(Function, Function));
  static_assert(!__is_convertible_to(PrivateCopy, PrivateCopy));
  static_assert(__is_convertible_to(X0<int>, X0<float>));
  static_assert(!__is_convertible_to(Abstract, Abstract));
}

namespace is_convertible_to_instantiate {
  // Make sure we don't try to instantiate the constructor.
  template<int x> class A { A(int) { int a[x]; } };
  int x = __is_convertible_to(int, A<-1>);
}

void is_trivial()
{
  static_assert(__is_trivial(int));
  static_assert(__is_trivial(Enum));
  static_assert(__is_trivial(POD));
  static_assert(__is_trivial(Int));
  static_assert(__is_trivial(IntAr));
  static_assert(__is_trivial(IntArNB));
  static_assert(__is_trivial(Statics));
  static_assert(__is_trivial(Empty));
  static_assert(__is_trivial(EmptyUnion));
  static_assert(__is_trivial(Union));
  static_assert(__is_trivial(Derives));
  static_assert(__is_trivial(DerivesAr));
  static_assert(__is_trivial(DerivesArNB));
  static_assert(__is_trivial(DerivesEmpty));
  static_assert(__is_trivial(HasFunc));
  static_assert(__is_trivial(HasOp));
  static_assert(__is_trivial(HasConv));
  static_assert(__is_trivial(HasAssign));
  static_assert(__is_trivial(HasAnonymousUnion));
  static_assert(__is_trivial(HasPriv));
  static_assert(__is_trivial(HasProt));
  static_assert(__is_trivial(DerivesHasPriv));
  static_assert(__is_trivial(DerivesHasProt));
  static_assert(__is_trivial(Vector));
  static_assert(__is_trivial(VectorExt));

  static_assert(!__is_trivial(HasCons));
  static_assert(!__is_trivial(HasCopyAssign));
  static_assert(!__is_trivial(HasMoveAssign));
  static_assert(!__is_trivial(HasDest));
  static_assert(!__is_trivial(HasRef));
  static_assert(!__is_trivial(HasNonPOD));
  static_assert(!__is_trivial(HasVirt));
  static_assert(!__is_trivial(DerivesHasCons));
  static_assert(!__is_trivial(DerivesHasCopyAssign));
  static_assert(!__is_trivial(DerivesHasMoveAssign));
  static_assert(!__is_trivial(DerivesHasDest));
  static_assert(!__is_trivial(DerivesHasRef));
  static_assert(!__is_trivial(DerivesHasVirt));
  static_assert(!__is_trivial(void));
  static_assert(!__is_trivial(cvoid));
}

template<typename T> struct TriviallyConstructibleTemplate {};

template<typename A, typename B, bool result = __is_assignable(A, B)>
static constexpr bool is_assignable_sfinae() { return result; }

void trivial_checks()
{
  static_assert(__is_trivially_copyable(int));
  static_assert(__is_trivially_copyable(Enum));
  static_assert(__is_trivially_copyable(POD));
  static_assert(__is_trivially_copyable(Int));
  static_assert(__is_trivially_copyable(IntAr));
  static_assert(__is_trivially_copyable(IntArNB));
  static_assert(__is_trivially_copyable(Statics));
  static_assert(__is_trivially_copyable(Empty));
  static_assert(__is_trivially_copyable(EmptyUnion));
  static_assert(__is_trivially_copyable(Union));
  static_assert(__is_trivially_copyable(Derives));
  static_assert(__is_trivially_copyable(DerivesAr));
  static_assert(__is_trivially_copyable(DerivesArNB));
  static_assert(__is_trivially_copyable(DerivesEmpty));
  static_assert(__is_trivially_copyable(HasFunc));
  static_assert(__is_trivially_copyable(HasOp));
  static_assert(__is_trivially_copyable(HasConv));
  static_assert(__is_trivially_copyable(HasAssign));
  static_assert(__is_trivially_copyable(HasAnonymousUnion));
  static_assert(__is_trivially_copyable(HasPriv));
  static_assert(__is_trivially_copyable(HasProt));
  static_assert(__is_trivially_copyable(DerivesHasPriv));
  static_assert(__is_trivially_copyable(DerivesHasProt));
  static_assert(__is_trivially_copyable(Vector));
  static_assert(__is_trivially_copyable(VectorExt));
  static_assert(__is_trivially_copyable(HasCons));
  static_assert(__is_trivially_copyable(HasRef));
  static_assert(__is_trivially_copyable(HasNonPOD));
  static_assert(__is_trivially_copyable(DerivesHasCons));
  static_assert(__is_trivially_copyable(DerivesHasRef));
  static_assert(__is_trivially_copyable(NonTrivialDefault));
  static_assert(__is_trivially_copyable(NonTrivialDefault[]));
  static_assert(__is_trivially_copyable(NonTrivialDefault[3]));

  static_assert(!__is_trivially_copyable(HasCopyAssign));
  static_assert(!__is_trivially_copyable(HasMoveAssign));
  static_assert(!__is_trivially_copyable(HasDest));
  static_assert(!__is_trivially_copyable(HasVirt));
  static_assert(!__is_trivially_copyable(DerivesHasCopyAssign));
  static_assert(!__is_trivially_copyable(DerivesHasMoveAssign));
  static_assert(!__is_trivially_copyable(DerivesHasDest));
  static_assert(!__is_trivially_copyable(DerivesHasVirt));
  static_assert(!__is_trivially_copyable(void));
  static_assert(!__is_trivially_copyable(cvoid));

  static_assert((__is_trivially_constructible(int)));
  static_assert((__is_trivially_constructible(int, int)));
  static_assert((__is_trivially_constructible(int, float)));
  static_assert((__is_trivially_constructible(int, int&)));
  static_assert((__is_trivially_constructible(int, const int&)));
  static_assert((__is_trivially_constructible(int, int)));
  static_assert((__is_trivially_constructible(HasCopyAssign, HasCopyAssign)));
  static_assert((__is_trivially_constructible(HasCopyAssign, const HasCopyAssign&)));
  static_assert((__is_trivially_constructible(HasCopyAssign, HasCopyAssign&&)));
  static_assert((__is_trivially_constructible(HasCopyAssign)));
  static_assert((__is_trivially_constructible(NonTrivialDefault,
                                            const NonTrivialDefault&)));
  static_assert((__is_trivially_constructible(NonTrivialDefault,
                                            NonTrivialDefault&&)));
  static_assert((__is_trivially_constructible(AllDefaulted)));
  static_assert((__is_trivially_constructible(AllDefaulted,
                                            const AllDefaulted &)));
  static_assert((__is_trivially_constructible(AllDefaulted,
                                            AllDefaulted &&)));

  static_assert(!(__is_trivially_constructible(int, int*)));
  static_assert(!(__is_trivially_constructible(NonTrivialDefault)));
  static_assert(!(__is_trivially_constructible(ThreeArgCtor, int*, char*, int&)));
  static_assert(!(__is_trivially_constructible(AllDeleted)));
  static_assert(!(__is_trivially_constructible(AllDeleted,
                                            const AllDeleted &)));
  static_assert(!(__is_trivially_constructible(AllDeleted,
                                            AllDeleted &&)));
  static_assert(!(__is_trivially_constructible(ExtDefaulted)));
  static_assert(!(__is_trivially_constructible(ExtDefaulted,
                                            const ExtDefaulted &)));
  static_assert(!(__is_trivially_constructible(ExtDefaulted,
                                            ExtDefaulted &&)));

  static_assert((__is_trivially_constructible(TriviallyConstructibleTemplate<int>)));
  static_assert(!(__is_trivially_constructible(class_forward))); // expected-error {{incomplete type 'class_forward' used in type trait expression}}
  static_assert(!(__is_trivially_constructible(class_forward[])));
  static_assert(!(__is_trivially_constructible(void)));

  static_assert((__is_trivially_assignable(int&, int)));
  static_assert((__is_trivially_assignable(int&, int&)));
  static_assert((__is_trivially_assignable(int&, int&&)));
  static_assert((__is_trivially_assignable(int&, const int&)));
  static_assert((__is_trivially_assignable(POD&, POD)));
  static_assert((__is_trivially_assignable(POD&, POD&)));
  static_assert((__is_trivially_assignable(POD&, POD&&)));
  static_assert((__is_trivially_assignable(POD&, const POD&)));
  static_assert((__is_trivially_assignable(int*&, int*)));
  static_assert((__is_trivially_assignable(AllDefaulted,
                                         const AllDefaulted &)));
  static_assert((__is_trivially_assignable(AllDefaulted,
                                         AllDefaulted &&)));

  static_assert(!(__is_trivially_assignable(int*&, float*)));
  static_assert(!(__is_trivially_assignable(HasCopyAssign&, HasCopyAssign)));
  static_assert(!(__is_trivially_assignable(HasCopyAssign&, HasCopyAssign&)));
  static_assert(!(__is_trivially_assignable(HasCopyAssign&, const HasCopyAssign&)));
  static_assert(!(__is_trivially_assignable(HasCopyAssign&, HasCopyAssign&&)));
  static_assert(!(__is_trivially_assignable(TrivialMoveButNotCopy&,
                                        TrivialMoveButNotCopy&)));
  static_assert(!(__is_trivially_assignable(TrivialMoveButNotCopy&,
                                        const TrivialMoveButNotCopy&)));
  static_assert(!(__is_trivially_assignable(AllDeleted,
                                         const AllDeleted &)));
  static_assert(!(__is_trivially_assignable(AllDeleted,
                                         AllDeleted &&)));
  static_assert(!(__is_trivially_assignable(ExtDefaulted,
                                         const ExtDefaulted &)));
  static_assert(!(__is_trivially_assignable(ExtDefaulted,
                                         ExtDefaulted &&)));

  static_assert((__is_trivially_assignable(HasDefaultTrivialCopyAssign&,
                                         HasDefaultTrivialCopyAssign&)));
  static_assert((__is_trivially_assignable(HasDefaultTrivialCopyAssign&,
                                       const HasDefaultTrivialCopyAssign&)));
  static_assert((__is_trivially_assignable(TrivialMoveButNotCopy&,
                                         TrivialMoveButNotCopy)));
  static_assert((__is_trivially_assignable(TrivialMoveButNotCopy&,
                                         TrivialMoveButNotCopy&&)));
  static_assert((__is_trivially_assignable(int&, int)));
  static_assert((__is_trivially_assignable(int&, int&)));
  static_assert((__is_trivially_assignable(int&, int&&)));
  static_assert((__is_trivially_assignable(int&, const int&)));
  static_assert((__is_trivially_assignable(POD&, POD)));
  static_assert((__is_trivially_assignable(POD&, POD&)));
  static_assert((__is_trivially_assignable(POD&, POD&&)));
  static_assert((__is_trivially_assignable(POD&, const POD&)));
  static_assert((__is_trivially_assignable(int*&, int*)));
  static_assert((__is_trivially_assignable(AllDefaulted,
                                         const AllDefaulted &)));
  static_assert((__is_trivially_assignable(AllDefaulted,
                                         AllDefaulted &&)));

  static_assert(!(__is_assignable(int *&, float *)));
  static_assert((__is_assignable(HasCopyAssign &, HasCopyAssign)));
  static_assert((__is_assignable(HasCopyAssign &, HasCopyAssign &)));
  static_assert((__is_assignable(HasCopyAssign &, const HasCopyAssign &)));
  static_assert((__is_assignable(HasCopyAssign &, HasCopyAssign &&)));
  static_assert((__is_assignable(TrivialMoveButNotCopy &,
                               TrivialMoveButNotCopy &)));
  static_assert((__is_assignable(TrivialMoveButNotCopy &,
                               const TrivialMoveButNotCopy &)));
  static_assert(!(__is_assignable(AllDeleted,
                               const AllDeleted &)));
  static_assert(!(__is_assignable(AllDeleted,
                               AllDeleted &&)));
  static_assert((__is_assignable(ExtDefaulted,
                               const ExtDefaulted &)));
  static_assert((__is_assignable(ExtDefaulted,
                               ExtDefaulted &&)));

  static_assert((__is_assignable(HasDefaultTrivialCopyAssign &,
                               HasDefaultTrivialCopyAssign &)));
  static_assert((__is_assignable(HasDefaultTrivialCopyAssign &,
                               const HasDefaultTrivialCopyAssign &)));
  static_assert((__is_assignable(TrivialMoveButNotCopy &,
                               TrivialMoveButNotCopy)));
  static_assert((__is_assignable(TrivialMoveButNotCopy &,
                               TrivialMoveButNotCopy &&)));

  static_assert(__is_assignable(ACompleteType, ACompleteType));
  static_assert(!__is_assignable(AnIncompleteType, AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_assignable(AnIncompleteType[], AnIncompleteType[]));
  static_assert(!__is_assignable(AnIncompleteType[1], AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_assignable(void, void));
  static_assert(!__is_assignable(const volatile void, const volatile void));

  static_assert(!__is_assignable(AllPrivate, AllPrivate));
  // Make sure we don't emit "'operator=' is a private member" in SFINAE context.
  static_assert(!is_assignable_sfinae<AllPrivate, AllPrivate>());
}

void constructible_checks() {
  static_assert(__is_constructible(HasNoThrowConstructorWithArgs));
  static_assert(!__is_nothrow_constructible(HasNoThrowConstructorWithArgs)); // MSVC doesn't look into default args and gets this wrong.

  static_assert(__is_constructible(HasNoThrowConstructorWithArgs, HasCons));
  static_assert(__is_nothrow_constructible(HasNoThrowConstructorWithArgs, HasCons));

  static_assert(__is_constructible(NonTrivialDefault));
  static_assert(!__is_nothrow_constructible(NonTrivialDefault));

  static_assert(__is_constructible(int));
  static_assert(__is_nothrow_constructible(int));

  static_assert(!__is_constructible(NonPOD));
  static_assert(!__is_nothrow_constructible(NonPOD));

  static_assert(__is_constructible(NonPOD, int));
  static_assert(!__is_nothrow_constructible(NonPOD, int));

  // PR19178
  static_assert(!__is_constructible(Abstract));
  static_assert(!__is_nothrow_constructible(Abstract));

  // PR20228
  static_assert(__is_constructible(VariadicCtor,
                                 int, int, int, int, int, int, int, int, int));

  // PR25513
  static_assert(!__is_constructible(int(int)));
  static_assert(__is_constructible(int const &, long));

  static_assert(__is_constructible(ACompleteType));
  static_assert(__is_nothrow_constructible(ACompleteType));
  static_assert(!__is_constructible(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_nothrow_constructible(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_constructible(AnIncompleteType[]));
  static_assert(!__is_nothrow_constructible(AnIncompleteType[]));
  static_assert(!__is_constructible(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_nothrow_constructible(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_constructible(void));
  static_assert(!__is_nothrow_constructible(void));
  static_assert(!__is_constructible(const volatile void));
  static_assert(!__is_nothrow_constructible(const volatile void));
}

// Instantiation of __is_trivially_constructible
template<typename T, typename ...Args>
struct is_trivially_constructible {
  static const bool value = __is_trivially_constructible(T, Args...);
};

void is_trivially_constructible_test() {
  static_assert((is_trivially_constructible<int>::value));
  static_assert((is_trivially_constructible<int, int>::value));
  static_assert((is_trivially_constructible<int, float>::value));
  static_assert((is_trivially_constructible<int, int&>::value));
  static_assert((is_trivially_constructible<int, const int&>::value));
  static_assert((is_trivially_constructible<int, int>::value));
  static_assert((is_trivially_constructible<HasCopyAssign, HasCopyAssign>::value));
  static_assert((is_trivially_constructible<HasCopyAssign, const HasCopyAssign&>::value));
  static_assert((is_trivially_constructible<HasCopyAssign, HasCopyAssign&&>::value));
  static_assert((is_trivially_constructible<HasCopyAssign>::value));
  static_assert((is_trivially_constructible<NonTrivialDefault,
                                            const NonTrivialDefault&>::value));
  static_assert((is_trivially_constructible<NonTrivialDefault,
                                            NonTrivialDefault&&>::value));

  static_assert(!(is_trivially_constructible<int, int*>::value));
  static_assert(!(is_trivially_constructible<NonTrivialDefault>::value));
  static_assert(!(is_trivially_constructible<ThreeArgCtor, int*, char*, int&>::value));
  static_assert(!(is_trivially_constructible<Abstract>::value)); // PR19178

  static_assert(__is_trivially_constructible(ACompleteType));
  static_assert(!__is_trivially_constructible(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_constructible(AnIncompleteType[]));
  static_assert(!__is_trivially_constructible(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_constructible(void));
  static_assert(!__is_trivially_constructible(const volatile void));
}

template <class T, class RefType = T &>
struct ConvertsToRef {
  operator RefType() const { return static_cast<RefType>(obj); }
  mutable T obj = 42;
};
template <class T, class RefType = T &>
class ConvertsToRefPrivate {
  operator RefType() const { return static_cast<RefType>(obj); }
  mutable T obj = 42;
};


void reference_binds_to_temporary_checks() {
  static_assert(!(__reference_binds_to_temporary(int &, int &)));
  static_assert(!(__reference_binds_to_temporary(int &, int &&)));

  static_assert(!(__reference_binds_to_temporary(int const &, int &)));
  static_assert(!(__reference_binds_to_temporary(int const &, int const &)));
  static_assert(!(__reference_binds_to_temporary(int const &, int &&)));

  static_assert(!(__reference_binds_to_temporary(int &, long &))); // doesn't construct
  static_assert((__reference_binds_to_temporary(int const &, long &)));
  static_assert((__reference_binds_to_temporary(int const &, long &&)));
  static_assert((__reference_binds_to_temporary(int &&, long &)));

  using LRef = ConvertsToRef<int, int &>;
  using RRef = ConvertsToRef<int, int &&>;
  using CLRef = ConvertsToRef<int, const int &>;
  using LongRef = ConvertsToRef<long, long &>;
  static_assert((__is_constructible(int &, LRef)));
  static_assert(!(__reference_binds_to_temporary(int &, LRef)));

  static_assert((__is_constructible(int &&, RRef)));
  static_assert(!(__reference_binds_to_temporary(int &&, RRef)));

  static_assert((__is_constructible(int const &, CLRef)));
  static_assert(!(__reference_binds_to_temporary(int &&, CLRef)));

  static_assert((__is_constructible(int const &, LongRef)));
  static_assert((__reference_binds_to_temporary(int const &, LongRef)));
  static_assert(!__reference_binds_to_temporary(int const &, ConvertsToRefPrivate<long, long &>));


  // Test that it doesn't accept non-reference types as input.
  static_assert(!(__reference_binds_to_temporary(int, long)));

  static_assert((__reference_binds_to_temporary(const int &, long)));

  // Test that function references are never considered bound to temporaries.
  static_assert(!__reference_binds_to_temporary(void(&)(), void()));
  static_assert(!__reference_binds_to_temporary(void(&&)(), void()));
}


struct ExplicitConversionRvalueRef {
    operator int();
    explicit operator int&&();
};

struct ExplicitConversionRef {
    operator int();
    explicit operator int&();
};

struct NonMovable {
  NonMovable(NonMovable&&) = delete;
};

struct ConvertsFromNonMovable {
  ConvertsFromNonMovable(NonMovable);
};

void reference_constructs_from_temporary_checks() {
  static_assert(!__reference_constructs_from_temporary(int &, int &));
  static_assert(!__reference_constructs_from_temporary(int &, int &&));

  static_assert(!__reference_constructs_from_temporary(int const &, int &));
  static_assert(!__reference_constructs_from_temporary(int const &, int const &));
  static_assert(!__reference_constructs_from_temporary(int const &, int &&));

  static_assert(!__reference_constructs_from_temporary(int &, long &)); // doesn't construct

  static_assert(__reference_constructs_from_temporary(int const &, long &));
  static_assert(__reference_constructs_from_temporary(int const &, long &&));
  static_assert(__reference_constructs_from_temporary(int &&, long &));

  using LRef = ConvertsToRef<int, int &>;
  using RRef = ConvertsToRef<int, int &&>;
  using CLRef = ConvertsToRef<int, const int &>;
  using LongRef = ConvertsToRef<long, long &>;
  static_assert(__is_constructible(int &, LRef));
  static_assert(!__reference_constructs_from_temporary(int &, LRef));

  static_assert(__is_constructible(int &&, RRef));
  static_assert(!__reference_constructs_from_temporary(int &&, RRef));

  static_assert(__is_constructible(int const &, CLRef));
  static_assert(!__reference_constructs_from_temporary(int &&, CLRef));

  static_assert(__is_constructible(int const &, LongRef));
  static_assert(__reference_constructs_from_temporary(int const &, LongRef));
  static_assert(!__reference_constructs_from_temporary(int const &, ConvertsToRefPrivate<long, long &>));


  // Test that it doesn't accept non-reference types as input.
  static_assert(!__reference_constructs_from_temporary(int, long));

  static_assert(__reference_constructs_from_temporary(const int &, long));

  // Test that function references are never considered bound to temporaries.
  static_assert(!__reference_constructs_from_temporary(void(&&)(), void()));
  static_assert(!__reference_constructs_from_temporary(void(&)(), void()));

  // LWG3819: reference_meows_from_temporary should not use is_meowible
  static_assert(__reference_constructs_from_temporary(ConvertsFromNonMovable&&, NonMovable) == __cplusplus >= 201703L);
  // For scalar types, cv-qualifications are dropped first for prvalues.
  static_assert(__reference_constructs_from_temporary(int&&, const int));
  static_assert(__reference_constructs_from_temporary(int&&, volatile int));

  // Additional checks
  static_assert(__reference_constructs_from_temporary(POD const&, Derives));
  static_assert(__reference_constructs_from_temporary(int&&, int));
  static_assert(__reference_constructs_from_temporary(const int&, int));
  static_assert(!__reference_constructs_from_temporary(int&&, int&&));
  static_assert(!__reference_constructs_from_temporary(const int&, int&&));
  static_assert(__reference_constructs_from_temporary(int&&, long&&));
  static_assert(__reference_constructs_from_temporary(int&&, long));


  static_assert(!__reference_constructs_from_temporary(int&, ExplicitConversionRef));
  static_assert(!__reference_constructs_from_temporary(const int&, ExplicitConversionRef));
  static_assert(!__reference_constructs_from_temporary(int&&, ExplicitConversionRvalueRef));


}

template<typename A, typename B, bool result = __reference_converts_from_temporary(A, B)>
static constexpr bool reference_converts_from_temporary_sfinae() { return result; }

void reference_converts_from_temporary_checks() {
  static_assert(!__reference_converts_from_temporary(int &, int &));
  static_assert(!__reference_converts_from_temporary(int &, int &&));

  static_assert(!__reference_converts_from_temporary(int const &, int &));
  static_assert(!__reference_converts_from_temporary(int const &, int const &));
  static_assert(!__reference_converts_from_temporary(int const &, int &&));

  static_assert(!__reference_converts_from_temporary(int &, long &)); // doesn't construct

  static_assert(__reference_converts_from_temporary(int const &, long &));
  static_assert(__reference_converts_from_temporary(int const &, long &&));
  static_assert(__reference_converts_from_temporary(int &&, long &));

  using LRef = ConvertsToRef<int, int &>;
  using RRef = ConvertsToRef<int, int &&>;
  using CLRef = ConvertsToRef<int, const int &>;
  using LongRef = ConvertsToRef<long, long &>;
  static_assert(__is_constructible(int &, LRef));
  static_assert(!__reference_converts_from_temporary(int &, LRef));

  static_assert(__is_constructible(int &&, RRef));
  static_assert(!__reference_converts_from_temporary(int &&, RRef));

  static_assert(__is_constructible(int const &, CLRef));
  static_assert(!__reference_converts_from_temporary(int &&, CLRef));

  static_assert(__is_constructible(int const &, LongRef));
  static_assert(__reference_converts_from_temporary(int const &, LongRef));
  static_assert(!__reference_converts_from_temporary(int const &, ConvertsToRefPrivate<long, long &>));


  // Test that it doesn't accept non-reference types as input.
  static_assert(!__reference_converts_from_temporary(int, long));

  static_assert(__reference_converts_from_temporary(const int &, long));

  // Test that function references are never considered bound to temporaries.
  static_assert(!__reference_converts_from_temporary(void(&)(), void()));
  static_assert(!__reference_converts_from_temporary(void(&&)(), void()));

  // LWG3819: reference_meows_from_temporary should not use is_meowible
  static_assert(__reference_converts_from_temporary(ConvertsFromNonMovable&&, NonMovable) == __cplusplus >= 201703L);
  // For scalar types, cv-qualifications are dropped first for prvalues.
  static_assert(__reference_converts_from_temporary(int&&, const int));
  static_assert(__reference_converts_from_temporary(int&&, volatile int));

  // Additional checks
  static_assert(__reference_converts_from_temporary(POD const&, Derives));
  static_assert(__reference_converts_from_temporary(int&&, int));
  static_assert(__reference_converts_from_temporary(const int&, int));
  static_assert(!__reference_converts_from_temporary(int&&, int&&));
  static_assert(!__reference_converts_from_temporary(const int&, int&&));
  static_assert(__reference_converts_from_temporary(int&&, long&&));
  static_assert(__reference_converts_from_temporary(int&&, long));

  static_assert(!__reference_converts_from_temporary(int&, ExplicitConversionRef));
  static_assert(__reference_converts_from_temporary(const int&, ExplicitConversionRef));
  static_assert(__reference_converts_from_temporary(int&&, ExplicitConversionRvalueRef));

  static_assert(!__reference_converts_from_temporary(AllPrivate, AllPrivate));
  // Make sure we don't emit "calling a private constructor" in SFINAE context.
  static_assert(!reference_converts_from_temporary_sfinae<AllPrivate, AllPrivate>());
}

void array_rank() {
  static_assert(__array_rank(IntAr) == 1);
  static_assert(__array_rank(ConstIntArAr) == 2);
}

void array_extent() {
  static_assert(__array_extent(IntAr, 0) == 10);
  static_assert(__array_extent(ConstIntArAr, 0) == 4);
  static_assert(__array_extent(ConstIntArAr, 1) == 10);
}

void is_destructible_test() {
  static_assert(__is_destructible(int));
  static_assert(__is_destructible(int[2]));
  static_assert(!__is_destructible(int[]));
  static_assert(!__is_destructible(void));
  static_assert(__is_destructible(int &));
  static_assert(__is_destructible(HasDest));
  static_assert(!__is_destructible(AllPrivate));
  static_assert(__is_destructible(SuperNonTrivialStruct));
  static_assert(__is_destructible(AllDefaulted));
  static_assert(!__is_destructible(AllDeleted));
  static_assert(__is_destructible(ThrowingDtor));
  static_assert(__is_destructible(NoThrowDtor));

  static_assert(__is_destructible(ACompleteType));
  static_assert(!__is_destructible(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_destructible(AnIncompleteType[]));
  static_assert(!__is_destructible(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_destructible(void));
  static_assert(!__is_destructible(const volatile void));
}

void is_nothrow_destructible_test() {
  static_assert(__is_nothrow_destructible(int));
  static_assert(__is_nothrow_destructible(int[2]));
  static_assert(!__is_nothrow_destructible(int[]));
  static_assert(!__is_nothrow_destructible(void));
  static_assert(__is_nothrow_destructible(int &));
  static_assert(__is_nothrow_destructible(HasDest));
  static_assert(!__is_nothrow_destructible(AllPrivate));
  static_assert(__is_nothrow_destructible(SuperNonTrivialStruct));
  static_assert(__is_nothrow_destructible(AllDefaulted));
  static_assert(!__is_nothrow_destructible(AllDeleted));
  static_assert(!__is_nothrow_destructible(ThrowingDtor));
  static_assert(__is_nothrow_destructible(NoExceptDtor));
  static_assert(__is_nothrow_destructible(NoThrowDtor));

  static_assert(__is_nothrow_destructible(ACompleteType));
  static_assert(!__is_nothrow_destructible(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_nothrow_destructible(AnIncompleteType[]));
  static_assert(!__is_nothrow_destructible(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_nothrow_destructible(void));
  static_assert(!__is_nothrow_destructible(const volatile void));
}

void is_trivially_destructible_test() {
  static_assert(__is_trivially_destructible(int));
  static_assert(__is_trivially_destructible(int[2]));
  static_assert(!__is_trivially_destructible(int[]));
  static_assert(!__is_trivially_destructible(void));
  static_assert(__is_trivially_destructible(int &));
  static_assert(!__is_trivially_destructible(HasDest));
  static_assert(!__is_trivially_destructible(AllPrivate));
  static_assert(!__is_trivially_destructible(SuperNonTrivialStruct));
  static_assert(__is_trivially_destructible(AllDefaulted));
  static_assert(!__is_trivially_destructible(AllDeleted));
  static_assert(!__is_trivially_destructible(ThrowingDtor));
  static_assert(!__is_trivially_destructible(NoThrowDtor));

  static_assert(__is_trivially_destructible(ACompleteType));
  static_assert(!__is_trivially_destructible(AnIncompleteType)); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_destructible(AnIncompleteType[]));
  static_assert(!__is_trivially_destructible(AnIncompleteType[1])); // expected-error {{incomplete type}}
  static_assert(!__is_trivially_destructible(void));
  static_assert(!__is_trivially_destructible(const volatile void));
}

static_assert(!__has_unique_object_representations(void), "void is never unique");
static_assert(!__has_unique_object_representations(const void), "void is never unique");
static_assert(!__has_unique_object_representations(volatile void), "void is never unique");
static_assert(!__has_unique_object_representations(const volatile void), "void is never unique");

static_assert(__has_unique_object_representations(int), "integrals are");
static_assert(__has_unique_object_representations(const int), "integrals are");
static_assert(__has_unique_object_representations(volatile int), "integrals are");
static_assert(__has_unique_object_representations(const volatile int), "integrals are");

static_assert(__has_unique_object_representations(void *), "as are pointers");
static_assert(__has_unique_object_representations(const void *), "as are pointers");
static_assert(__has_unique_object_representations(volatile void *), "are pointers");
static_assert(__has_unique_object_representations(const volatile void *), "as are pointers");

static_assert(__has_unique_object_representations(int *), "as are pointers");
static_assert(__has_unique_object_representations(const int *), "as are pointers");
static_assert(__has_unique_object_representations(volatile int *), "as are pointers");
static_assert(__has_unique_object_representations(const volatile int *), "as are pointers");

class C {};
using FP = int (*)(int);
using PMF = int (C::*)(int);
using PMD = int C::*;

static_assert(__has_unique_object_representations(FP), "even function pointers");
static_assert(__has_unique_object_representations(const FP), "even function pointers");
static_assert(__has_unique_object_representations(volatile FP), "even function pointers");
static_assert(__has_unique_object_representations(const volatile FP), "even function pointers");

static_assert(__has_unique_object_representations(PMF), "and pointer to members");
static_assert(__has_unique_object_representations(const PMF), "and pointer to members");
static_assert(__has_unique_object_representations(volatile PMF), "and pointer to members");
static_assert(__has_unique_object_representations(const volatile PMF), "and pointer to members");

static_assert(__has_unique_object_representations(PMD), "and pointer to members");
static_assert(__has_unique_object_representations(const PMD), "and pointer to members");
static_assert(__has_unique_object_representations(volatile PMD), "and pointer to members");
static_assert(__has_unique_object_representations(const volatile PMD), "and pointer to members");

static_assert(__has_unique_object_representations(bool), "yes, all integral types");
static_assert(__has_unique_object_representations(char), "yes, all integral types");
static_assert(__has_unique_object_representations(signed char), "yes, all integral types");
static_assert(__has_unique_object_representations(unsigned char), "yes, all integral types");
static_assert(__has_unique_object_representations(short), "yes, all integral types");
static_assert(__has_unique_object_representations(unsigned short), "yes, all integral types");
static_assert(__has_unique_object_representations(int), "yes, all integral types");
static_assert(__has_unique_object_representations(unsigned int), "yes, all integral types");
static_assert(__has_unique_object_representations(long), "yes, all integral types");
static_assert(__has_unique_object_representations(unsigned long), "yes, all integral types");
static_assert(__has_unique_object_representations(long long), "yes, all integral types");
static_assert(__has_unique_object_representations(unsigned long long), "yes, all integral types");
static_assert(__has_unique_object_representations(wchar_t), "yes, all integral types");
static_assert(__has_unique_object_representations(char16_t), "yes, all integral types");
static_assert(__has_unique_object_representations(char32_t), "yes, all integral types");

static_assert(!__has_unique_object_representations(void), "but not void!");
static_assert(!__has_unique_object_representations(decltype(nullptr)), "or nullptr_t");
static_assert(!__has_unique_object_representations(float), "definitely not Floating Point");
static_assert(!__has_unique_object_representations(double), "definitely not Floating Point");
static_assert(!__has_unique_object_representations(long double), "definitely not Floating Point");


static_assert(!__has_unique_object_representations(AnIncompleteType[]));
//expected-error@-1 {{incomplete type 'AnIncompleteType' used in type trait expression}}
static_assert(!__has_unique_object_representations(AnIncompleteType[][1]));
//expected-error@-1 {{incomplete type 'AnIncompleteType' used in type trait expression}}
static_assert(!__has_unique_object_representations(AnIncompleteType[1]));
//expected-error@-1 {{incomplete type 'AnIncompleteType' used in type trait expression}}
static_assert(!__has_unique_object_representations(AnIncompleteType));
//expected-error@-1 {{incomplete type 'AnIncompleteType' used in type trait expression}}

struct NoPadding {
  int a;
  int b;
};

static_assert(__has_unique_object_representations(NoPadding), "types without padding are");

struct InheritsFromNoPadding : NoPadding {
  int c;
  int d;
};

static_assert(__has_unique_object_representations(InheritsFromNoPadding), "types without padding are");

struct VirtuallyInheritsFromNoPadding : virtual NoPadding {
  int c;
  int d;
};

static_assert(!__has_unique_object_representations(VirtuallyInheritsFromNoPadding), "No virtual inheritance");

struct Padding {
  char a;
  int b;
};

//static_assert(!__has_unique_object_representations(Padding), "but not with padding");

struct InheritsFromPadding : Padding {
  int c;
  int d;
};

static_assert(!__has_unique_object_representations(InheritsFromPadding), "or its subclasses");

struct TailPadding {
  int a;
  char b;
};

static_assert(!__has_unique_object_representations(TailPadding), "even at the end");

struct TinyStruct {
  char a;
};

static_assert(__has_unique_object_representations(TinyStruct), "Should be no padding");

struct InheritsFromTinyStruct : TinyStruct {
  int b;
};

static_assert(!__has_unique_object_representations(InheritsFromTinyStruct), "Inherit causes padding");

union NoPaddingUnion {
  int a;
  unsigned int b;
};

static_assert(__has_unique_object_representations(NoPaddingUnion), "unions follow the same rules as structs");

union PaddingUnion {
  int a;
  long long b;
};

static_assert(!__has_unique_object_representations(PaddingUnion), "unions follow the same rules as structs");

struct NotTriviallyCopyable {
  int x;
  NotTriviallyCopyable(const NotTriviallyCopyable &) {}
};

static_assert(!__has_unique_object_representations(NotTriviallyCopyable), "must be trivially copyable");

struct HasNonUniqueMember {
  float x;
};

static_assert(!__has_unique_object_representations(HasNonUniqueMember), "all members must be unique");

enum ExampleEnum { xExample,
                   yExample };
enum LLEnum : long long { xLongExample,
                          yLongExample };

static_assert(__has_unique_object_representations(ExampleEnum), "Enums are integrals, so unique!");
static_assert(__has_unique_object_representations(LLEnum), "Enums are integrals, so unique!");

enum class ExampleEnumClass { xExample,
                              yExample };
enum class LLEnumClass : long long { xLongExample,
                                     yLongExample };

static_assert(__has_unique_object_representations(ExampleEnumClass), "Enums are integrals, so unique!");
static_assert(__has_unique_object_representations(LLEnumClass), "Enums are integrals, so unique!");

// because references aren't trivially copyable.
static_assert(!__has_unique_object_representations(int &), "No references!");
static_assert(!__has_unique_object_representations(const int &), "No references!");
static_assert(!__has_unique_object_representations(volatile int &), "No references!");
static_assert(!__has_unique_object_representations(const volatile int &), "No references!");
static_assert(!__has_unique_object_representations(Empty), "No empty types!");
static_assert(!__has_unique_object_representations(EmptyUnion), "No empty types!");

class Compressed : Empty {
  int x;
};

static_assert(__has_unique_object_representations(Compressed), "But inheriting from one is ok");

class EmptyInheritor : Compressed {};

static_assert(__has_unique_object_representations(EmptyInheritor), "As long as the base has items, empty is ok");

class Dynamic {
  virtual void A();
  int i;
};

static_assert(!__has_unique_object_representations(Dynamic), "Dynamic types are not valid");

class InheritsDynamic : Dynamic {
  int j;
};

static_assert(!__has_unique_object_representations(InheritsDynamic), "Dynamic types are not valid");

static_assert(__has_unique_object_representations(int[42]), "Arrays are fine, as long as their value type is");
static_assert(__has_unique_object_representations(int[]), "Arrays are fine, as long as their value type is");
static_assert(__has_unique_object_representations(int[][42]), "Arrays are fine, as long as their value type is");
static_assert(!__has_unique_object_representations(double[42]), "So no array of doubles!");
static_assert(!__has_unique_object_representations(double[]), "So no array of doubles!");
static_assert(!__has_unique_object_representations(double[][42]), "So no array of doubles!");

struct __attribute__((aligned(16))) WeirdAlignment {
  int i;
};
union __attribute__((aligned(16))) WeirdAlignmentUnion {
  int i;
};
static_assert(!__has_unique_object_representations(WeirdAlignment), "Alignment causes padding");
static_assert(!__has_unique_object_representations(WeirdAlignmentUnion), "Alignment causes padding");
static_assert(!__has_unique_object_representations(WeirdAlignment[42]), "Also no arrays that have padding");

struct __attribute__((packed)) PackedNoPadding1 {
  short i;
  int j;
};
struct __attribute__((packed)) PackedNoPadding2 {
  int j;
  short i;
};
static_assert(__has_unique_object_representations(PackedNoPadding1), "Packed structs have no padding");
static_assert(__has_unique_object_representations(PackedNoPadding2), "Packed structs have no padding");

static_assert(!__has_unique_object_representations(int(int)), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) const), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) volatile), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) const volatile), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) const &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) volatile &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) const volatile &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) &&), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) const &&), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) volatile &&), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int) const volatile &&), "Functions are not unique");

static_assert(!__has_unique_object_representations(int(int, ...)), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) const), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) volatile), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) const volatile), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) const &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) volatile &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) const volatile &), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) &&), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) const &&), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) volatile &&), "Functions are not unique");
static_assert(!__has_unique_object_representations(int(int, ...) const volatile &&), "Functions are not unique");

void foo(){
  static auto lambda = []() {};
  static_assert(!__has_unique_object_representations(decltype(lambda)), "Lambdas follow struct rules");
  int i;
  static auto lambda2 = [i]() {};
  static_assert(__has_unique_object_representations(decltype(lambda2)), "Lambdas follow struct rules");
}

struct PaddedBitfield {
  char c : 6;
  char d : 1;
};

struct UnPaddedBitfield {
  char c : 6;
  char d : 2;
};

struct AlignedPaddedBitfield {
  char c : 6;
  __attribute__((aligned(1)))
  char d : 2;
};

struct UnnamedBitfield {
  int named : 8;
  int : 24;
};

struct __attribute__((packed)) UnnamedBitfieldPacked {
  int named : 8;
  int : 24;
};

struct UnnamedEmptyBitfield {
  int named;
  int : 0;
};

struct UnnamedEmptyBitfieldSplit {
  short named;
  int : 0;
  short also_named;
};

static_assert(!__has_unique_object_representations(PaddedBitfield), "Bitfield padding");
static_assert(__has_unique_object_representations(UnPaddedBitfield), "Bitfield padding");
static_assert(!__has_unique_object_representations(AlignedPaddedBitfield), "Bitfield padding");
static_assert(!__has_unique_object_representations(UnnamedBitfield), "Bitfield padding");
static_assert(!__has_unique_object_representations(UnnamedBitfieldPacked), "Bitfield padding");
static_assert(__has_unique_object_representations(UnnamedEmptyBitfield), "Bitfield padding");
static_assert(sizeof(UnnamedEmptyBitfieldSplit) != (sizeof(short) * 2), "Wrong size");
static_assert(!__has_unique_object_representations(UnnamedEmptyBitfieldSplit), "Bitfield padding");

struct BoolBitfield {
  bool b : 8;
};

static_assert(__has_unique_object_representations(BoolBitfield), "Bitfield bool");

struct BoolBitfield2 {
  bool b : 16;
};

static_assert(!__has_unique_object_representations(BoolBitfield2), "Bitfield bool");

struct GreaterSizeBitfield {
  //expected-warning@+1 {{width of bit-field 'n'}}
  int n : 1024;
};

static_assert(sizeof(GreaterSizeBitfield) == 128, "Bitfield Size");
static_assert(!__has_unique_object_representations(GreaterSizeBitfield), "Bitfield padding");

struct StructWithRef {
  int &I;
};

static_assert(__has_unique_object_representations(StructWithRef), "References are still unique");

struct NotUniqueBecauseTailPadding {
  int &r;
  char a;
};
struct CanBeUniqueIfNoPadding : NotUniqueBecauseTailPadding {
  char b[7];
};

static_assert(!__has_unique_object_representations(NotUniqueBecauseTailPadding),
              "non trivial");
// Can be unique on Itanium, since the is child class' data is 'folded' into the
// parent's tail padding.
static_assert(sizeof(CanBeUniqueIfNoPadding) != 16 ||
              __has_unique_object_representations(CanBeUniqueIfNoPadding),
              "inherit from std layout");

namespace ErrorType {
  struct S; //expected-note{{forward declaration of 'ErrorType::S'}}

  struct T {
        S t; //expected-error{{field has incomplete type 'S'}}
  };
  bool b = __has_unique_object_representations(T);
};

static_assert(!__has_unique_object_representations(_BitInt(7)), "BitInt:");
static_assert(__has_unique_object_representations(_BitInt(8)), "BitInt:");
static_assert(!__has_unique_object_representations(_BitInt(127)), "BitInt:");
static_assert(__has_unique_object_representations(_BitInt(128)), "BitInt:");

namespace GH95311 {

template <int>
class Foo {
  int x;
};
static_assert(__has_unique_object_representations(Foo<0>[]));
class Bar; // expected-note {{forward declaration of 'GH95311::Bar'}}
static_assert(__has_unique_object_representations(Bar[])); // expected-error {{incomplete type}}

}

namespace PR46209 {
  // Foo has both a trivial assignment operator and a non-trivial one.
  struct Foo {
    Foo &operator=(const Foo &) & { return *this; }
    Foo &operator=(const Foo &) && = default;
  };

  // Bar's copy assignment calls Foo's non-trivial assignment.
  struct Bar {
    Foo foo;
  };

  static_assert(!__is_trivially_assignable(Foo &, const Foo &));
  static_assert(!__is_trivially_assignable(Bar &, const Bar &));

  // Foo2 has both a trivial assignment operator and a non-trivial one.
  struct Foo2 {
    Foo2 &operator=(const Foo2 &) & = default;
    Foo2 &operator=(const Foo2 &) && { return *this; }
  };

  // Bar2's copy assignment calls Foo2's trivial assignment.
  struct Bar2 {
    Foo2 foo;
  };

  static_assert(__is_trivially_assignable(Foo2 &, const Foo2 &));
  static_assert(__is_trivially_assignable(Bar2 &, const Bar2 &));
}

namespace ConstClass {
  struct A {
    A &operator=(const A&) = default;
  };
  struct B {
    const A a;
  };
  static_assert(!__is_trivially_assignable(B&, const B&));
}

namespace type_trait_expr_numargs_overflow {
// Make sure that TypeTraitExpr can store 16 bits worth of arguments.
#define T4(X) X,X,X,X
#define T16(X) T4(X),T4(X),T4(X),T4(X)
#define T64(X) T16(X),T16(X),T16(X),T16(X)
#define T256(X) T64(X),T64(X),T64(X),T64(X)
#define T1024(X) T256(X),T256(X),T256(X),T256(X)
#define T4096(X) T1024(X),T1024(X),T1024(X),T1024(X)
#define T16384(X) T4096(X),T4096(X),T4096(X),T4096(X)
#define T32768(X) T16384(X),T16384(X)
void test() { (void) __is_constructible(int, T32768(int)); }
#undef T4
#undef T16
#undef T64
#undef T256
#undef T1024
#undef T4096
#undef T16384
#undef T32768
} // namespace type_trait_expr_numargs_overflow

namespace is_trivially_relocatable {

static_assert(!__is_trivially_relocatable(void));
static_assert(__is_trivially_relocatable(int));
static_assert(__is_trivially_relocatable(int[]));
static_assert(__is_trivially_relocatable(const int));
static_assert(__is_trivially_relocatable(volatile int));

enum Enum {};
static_assert(__is_trivially_relocatable(Enum));
static_assert(__is_trivially_relocatable(Enum[]));

union Union {int x;};
static_assert(__is_trivially_relocatable(Union));
static_assert(__is_trivially_relocatable(Union[]));

struct Trivial {};
static_assert(__is_trivially_relocatable(Trivial));
static_assert(__is_trivially_relocatable(const Trivial));
static_assert(__is_trivially_relocatable(volatile Trivial));

static_assert(__is_trivially_relocatable(Trivial[]));
static_assert(__is_trivially_relocatable(const Trivial[]));
static_assert(__is_trivially_relocatable(volatile Trivial[]));

static_assert(__is_trivially_relocatable(int[10]));
static_assert(__is_trivially_relocatable(const int[10]));
static_assert(__is_trivially_relocatable(volatile int[10]));

static_assert(__is_trivially_relocatable(int[10][10]));
static_assert(__is_trivially_relocatable(const int[10][10]));
static_assert(__is_trivially_relocatable(volatile int[10][10]));

static_assert(__is_trivially_relocatable(int[]));
static_assert(__is_trivially_relocatable(const int[]));
static_assert(__is_trivially_relocatable(volatile int[]));

static_assert(__is_trivially_relocatable(int[][10]));
static_assert(__is_trivially_relocatable(const int[][10]));
static_assert(__is_trivially_relocatable(volatile int[][10]));

struct Incomplete; // expected-note {{forward declaration of 'is_trivially_relocatable::Incomplete'}}
bool unused = __is_trivially_relocatable(Incomplete); // expected-error {{incomplete type}}

struct NontrivialDtor {
  ~NontrivialDtor() {}
};
static_assert(!__is_trivially_relocatable(NontrivialDtor));
static_assert(!__is_trivially_relocatable(NontrivialDtor[]));
static_assert(!__is_trivially_relocatable(const NontrivialDtor));
static_assert(!__is_trivially_relocatable(volatile NontrivialDtor));

struct NontrivialCopyCtor {
  NontrivialCopyCtor(const NontrivialCopyCtor&) {}
};
static_assert(!__is_trivially_relocatable(NontrivialCopyCtor));
static_assert(!__is_trivially_relocatable(NontrivialCopyCtor[]));

struct NontrivialMoveCtor {
  NontrivialMoveCtor(NontrivialMoveCtor&&) {}
};
static_assert(!__is_trivially_relocatable(NontrivialMoveCtor));
static_assert(!__is_trivially_relocatable(NontrivialMoveCtor[]));

struct [[clang::trivial_abi]] TrivialAbiNontrivialDtor {
  ~TrivialAbiNontrivialDtor() {}
};
static_assert(__is_trivially_relocatable(TrivialAbiNontrivialDtor));
static_assert(__is_trivially_relocatable(TrivialAbiNontrivialDtor[]));
static_assert(__is_trivially_relocatable(const TrivialAbiNontrivialDtor));
static_assert(__is_trivially_relocatable(volatile TrivialAbiNontrivialDtor));

struct [[clang::trivial_abi]] TrivialAbiNontrivialCopyCtor {
  TrivialAbiNontrivialCopyCtor(const TrivialAbiNontrivialCopyCtor&) {}
};
static_assert(__is_trivially_relocatable(TrivialAbiNontrivialCopyCtor));
static_assert(__is_trivially_relocatable(TrivialAbiNontrivialCopyCtor[]));
static_assert(__is_trivially_relocatable(const TrivialAbiNontrivialCopyCtor));
static_assert(__is_trivially_relocatable(volatile TrivialAbiNontrivialCopyCtor));

// A more complete set of tests for the behavior of trivial_abi can be found in
// clang/test/SemaCXX/attr-trivial-abi.cpp
struct [[clang::trivial_abi]] TrivialAbiNontrivialMoveCtor {
  TrivialAbiNontrivialMoveCtor(TrivialAbiNontrivialMoveCtor&&) {}
};
static_assert(__is_trivially_relocatable(TrivialAbiNontrivialMoveCtor));
static_assert(__is_trivially_relocatable(TrivialAbiNontrivialMoveCtor[]));
static_assert(__is_trivially_relocatable(const TrivialAbiNontrivialMoveCtor));
static_assert(__is_trivially_relocatable(volatile TrivialAbiNontrivialMoveCtor));

} // namespace is_trivially_relocatable

namespace is_trivially_equality_comparable {
struct ForwardDeclared; // expected-note {{forward declaration of 'is_trivially_equality_comparable::ForwardDeclared'}}
static_assert(!__is_trivially_equality_comparable(ForwardDeclared)); // expected-error {{incomplete type 'ForwardDeclared' used in type trait expression}}

static_assert(!__is_trivially_equality_comparable(void));
static_assert(__is_trivially_equality_comparable(int));
static_assert(!__is_trivially_equality_comparable(int[]));
static_assert(!__is_trivially_equality_comparable(int[3]));
static_assert(!__is_trivially_equality_comparable(float));
static_assert(!__is_trivially_equality_comparable(double));
static_assert(!__is_trivially_equality_comparable(long double));

struct NonTriviallyEqualityComparableNoComparator {
  int i;
  int j;
};
static_assert(!__is_trivially_equality_comparable(NonTriviallyEqualityComparableNoComparator));

struct NonTriviallyEqualityComparableConvertibleToBuiltin {
  int i;
  operator unsigned() const;
};
static_assert(!__is_trivially_equality_comparable(NonTriviallyEqualityComparableConvertibleToBuiltin));

struct NonTriviallyEqualityComparableNonDefaultedComparator {
  int i;
  int j;
  bool operator==(const NonTriviallyEqualityComparableNonDefaultedComparator&);
};
static_assert(!__is_trivially_equality_comparable(NonTriviallyEqualityComparableNonDefaultedComparator));

#if __cplusplus >= 202002L

struct TriviallyEqualityComparable {
  int i;
  int j;

  void func();
  bool operator==(int) const { return false; }

  bool operator==(const TriviallyEqualityComparable&) const = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparable));

struct TriviallyEqualityComparableContainsArray {
  int a[4];

  bool operator==(const TriviallyEqualityComparableContainsArray&) const = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableContainsArray));

struct TriviallyEqualityComparableContainsMultiDimensionArray {
  int a[4][4];

  bool operator==(const TriviallyEqualityComparableContainsMultiDimensionArray&) const = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableContainsMultiDimensionArray));

auto GetNonCapturingLambda() { return [](){ return 42; }; }

struct TriviallyEqualityComparableContainsLambda {
  [[no_unique_address]] decltype(GetNonCapturingLambda()) l;
  int i;

  bool operator==(const TriviallyEqualityComparableContainsLambda&) const = default;
};
static_assert(!__is_trivially_equality_comparable(decltype(GetNonCapturingLambda()))); // padding
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableContainsLambda));

struct TriviallyEqualityComparableNonTriviallyCopyable {
  TriviallyEqualityComparableNonTriviallyCopyable(const TriviallyEqualityComparableNonTriviallyCopyable&);
  ~TriviallyEqualityComparableNonTriviallyCopyable();
  bool operator==(const TriviallyEqualityComparableNonTriviallyCopyable&) const = default;
  int i;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableNonTriviallyCopyable));

struct NotTriviallyEqualityComparableHasPadding {
  short i;
  int j;

  bool operator==(const NotTriviallyEqualityComparableHasPadding&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasPadding));

struct NotTriviallyEqualityComparableHasFloat {
  float i;
  int j;

  bool operator==(const NotTriviallyEqualityComparableHasFloat&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasFloat));

struct NotTriviallyEqualityComparableHasTailPadding {
  int i;
  char j;

  bool operator==(const NotTriviallyEqualityComparableHasTailPadding&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasTailPadding));

struct NotTriviallyEqualityComparableBase : NotTriviallyEqualityComparableHasTailPadding {
  char j;

  bool operator==(const NotTriviallyEqualityComparableBase&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableBase));

class TriviallyEqualityComparablePaddedOutBase {
  int i;
  char c;

public:
  bool operator==(const TriviallyEqualityComparablePaddedOutBase&) const = default;
};
static_assert(!__is_trivially_equality_comparable(TriviallyEqualityComparablePaddedOutBase));

struct TriviallyEqualityComparablePaddedOut : TriviallyEqualityComparablePaddedOutBase {
  char j[3];

  bool operator==(const TriviallyEqualityComparablePaddedOut&) const = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparablePaddedOut));

struct TriviallyEqualityComparable1 {
  char i;

  bool operator==(const TriviallyEqualityComparable1&) const = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparable1));

struct TriviallyEqualityComparable2 {
  int i;

  bool operator==(const TriviallyEqualityComparable2&) const = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparable2));

struct NotTriviallyEqualityComparableTriviallyEqualityComparableBases
    : TriviallyEqualityComparable1, TriviallyEqualityComparable2 {
  bool operator==(const NotTriviallyEqualityComparableTriviallyEqualityComparableBases&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableTriviallyEqualityComparableBases));

struct NotTriviallyEqualityComparableBitfield {
  int i : 1;

  bool operator==(const NotTriviallyEqualityComparableBitfield&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableBitfield));

// TODO: This is trivially equality comparable
struct NotTriviallyEqualityComparableBitfieldFilled {
  char i : __CHAR_BIT__;

  bool operator==(const NotTriviallyEqualityComparableBitfieldFilled&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableBitfield));

union U {
  int i;

  bool operator==(const U&) const = default;
};

struct NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion {
  U u;

  bool operator==(const NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion));

struct NotTriviallyEqualityComparableExplicitlyDeleted {
  int i;

  bool operator==(const NotTriviallyEqualityComparableExplicitlyDeleted&) const = delete;
};

struct NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct {
  NotTriviallyEqualityComparableExplicitlyDeleted u;

  bool operator==(const NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct));

struct NotTriviallyEqualityComparableHasReferenceMember {
  int& i;

  bool operator==(const NotTriviallyEqualityComparableHasReferenceMember&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasReferenceMember));

struct NotTriviallyEqualityComparableNonTriviallyComparableBaseBase {
  int i;

  bool operator==(const NotTriviallyEqualityComparableNonTriviallyComparableBaseBase&) const {
    return true;
  }
};

struct NotTriviallyEqualityComparableNonTriviallyComparableBase : NotTriviallyEqualityComparableNonTriviallyComparableBaseBase {
  int i;

  bool operator==(const NotTriviallyEqualityComparableNonTriviallyComparableBase&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableNonTriviallyComparableBase));

enum E {
  a,
  b
};
bool operator==(E, E) { return false; }
static_assert(!__is_trivially_equality_comparable(E));

struct NotTriviallyEqualityComparableHasEnum {
  E e;
  bool operator==(const NotTriviallyEqualityComparableHasEnum&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasEnum));

struct NotTriviallyEqualityComparableNonTriviallyEqualityComparableArrs {
  E e[1];

  bool operator==(const NotTriviallyEqualityComparableNonTriviallyEqualityComparableArrs&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableNonTriviallyEqualityComparableArrs));

struct NotTriviallyEqualityComparableNonTriviallyEqualityComparableArrs2 {
  E e[1][1];

  bool operator==(const NotTriviallyEqualityComparableNonTriviallyEqualityComparableArrs2&) const = default;
};

static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableNonTriviallyEqualityComparableArrs2));

struct NotTriviallyEqualityComparablePrivateComparison {
  int i;

private:
  bool operator==(const NotTriviallyEqualityComparablePrivateComparison&) const = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparablePrivateComparison));

template<typename T, bool result = __is_trivially_equality_comparable(T)>
static constexpr bool is_trivially_equality_comparable_sfinae() { return result; }

// Make sure we don't emit "'operator==' is a private member" in SFINAE context.
static_assert(!is_trivially_equality_comparable_sfinae<NotTriviallyEqualityComparablePrivateComparison>());

template<bool B>
struct MaybeTriviallyEqualityComparable {
    int i;
    bool operator==(const MaybeTriviallyEqualityComparable&) const requires B = default;
    bool operator==(const MaybeTriviallyEqualityComparable& rhs) const { return (i % 3) == (rhs.i % 3); }
};
static_assert(__is_trivially_equality_comparable(MaybeTriviallyEqualityComparable<true>));
static_assert(!__is_trivially_equality_comparable(MaybeTriviallyEqualityComparable<false>));

struct NotTriviallyEqualityComparableMoreConstrainedExternalOp {
  int i;
  bool operator==(const NotTriviallyEqualityComparableMoreConstrainedExternalOp&) const = default;
};

bool operator==(const NotTriviallyEqualityComparableMoreConstrainedExternalOp&,
                const NotTriviallyEqualityComparableMoreConstrainedExternalOp&) __attribute__((enable_if(true, ""))) {}

static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableMoreConstrainedExternalOp));

struct TriviallyEqualityComparableExternalDefaultedOp {
  int i;
  friend bool operator==(TriviallyEqualityComparableExternalDefaultedOp, TriviallyEqualityComparableExternalDefaultedOp);
};
bool operator==(TriviallyEqualityComparableExternalDefaultedOp, TriviallyEqualityComparableExternalDefaultedOp) = default;

static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableExternalDefaultedOp));

struct EqualityComparableBase {
  bool operator==(const EqualityComparableBase&) const = default;
};

struct ComparingBaseOnly : EqualityComparableBase {
  int j_ = 0;
};
static_assert(!__is_trivially_equality_comparable(ComparingBaseOnly));

template <class>
class Template {};

// Make sure we don't crash when instantiating a type
static_assert(!__is_trivially_equality_comparable(Template<Template<int>>));


struct S operator==(S, S);

template <class> struct basic_string_view {};

struct basic_string {
  operator basic_string_view<int>() const;
};

template <class T>
const bool is_trivially_equality_comparable = __is_trivially_equality_comparable(T);

template <int = is_trivially_equality_comparable<basic_string> >
void find();

void func() { find(); }


namespace hidden_friend {

struct TriviallyEqualityComparable {
  int i;
  int j;

  void func();
  bool operator==(int) const { return false; }

  friend bool operator==(const TriviallyEqualityComparable&, const TriviallyEqualityComparable&) = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparable));

struct TriviallyEqualityComparableNonTriviallyCopyable {
  TriviallyEqualityComparableNonTriviallyCopyable(const TriviallyEqualityComparableNonTriviallyCopyable&);
  ~TriviallyEqualityComparableNonTriviallyCopyable();
  friend bool operator==(const TriviallyEqualityComparableNonTriviallyCopyable&, const TriviallyEqualityComparableNonTriviallyCopyable&) = default;
  int i;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableNonTriviallyCopyable));

struct NotTriviallyEqualityComparableHasPadding {
  short i;
  int j;

  friend bool operator==(const NotTriviallyEqualityComparableHasPadding&, const NotTriviallyEqualityComparableHasPadding&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasPadding));

struct NotTriviallyEqualityComparableHasFloat {
  float i;
  int j;

  friend bool operator==(const NotTriviallyEqualityComparableHasFloat&, const NotTriviallyEqualityComparableHasFloat&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasFloat));

struct NotTriviallyEqualityComparableHasTailPadding {
  int i;
  char j;

  friend bool operator==(const NotTriviallyEqualityComparableHasTailPadding&, const NotTriviallyEqualityComparableHasTailPadding&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasTailPadding));

struct NotTriviallyEqualityComparableBase : NotTriviallyEqualityComparableHasTailPadding {
  char j;

  friend bool operator==(const NotTriviallyEqualityComparableBase&, const NotTriviallyEqualityComparableBase&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableBase));

class TriviallyEqualityComparablePaddedOutBase {
  int i;
  char c;

public:
  friend bool operator==(const TriviallyEqualityComparablePaddedOutBase&, const TriviallyEqualityComparablePaddedOutBase&) = default;
};
static_assert(!__is_trivially_equality_comparable(TriviallyEqualityComparablePaddedOutBase));

struct TriviallyEqualityComparablePaddedOut : TriviallyEqualityComparablePaddedOutBase {
  char j[3];

  friend bool operator==(const TriviallyEqualityComparablePaddedOut&, const TriviallyEqualityComparablePaddedOut&) = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparablePaddedOut));

struct TriviallyEqualityComparable1 {
  char i;

  friend bool operator==(const TriviallyEqualityComparable1&, const TriviallyEqualityComparable1&) = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparable1));

struct TriviallyEqualityComparable2 {
  int i;

  friend bool operator==(const TriviallyEqualityComparable2&, const TriviallyEqualityComparable2&) = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparable2));

struct NotTriviallyEqualityComparableTriviallyEqualityComparableBases
    : TriviallyEqualityComparable1, TriviallyEqualityComparable2 {
  friend bool operator==(const NotTriviallyEqualityComparableTriviallyEqualityComparableBases&, const NotTriviallyEqualityComparableTriviallyEqualityComparableBases&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableTriviallyEqualityComparableBases));

struct NotTriviallyEqualityComparableBitfield {
  int i : 1;

  friend bool operator==(const NotTriviallyEqualityComparableBitfield&, const NotTriviallyEqualityComparableBitfield&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableBitfield));

// TODO: This is trivially equality comparable
struct NotTriviallyEqualityComparableBitfieldFilled {
  char i : __CHAR_BIT__;

  friend bool operator==(const NotTriviallyEqualityComparableBitfieldFilled&, const NotTriviallyEqualityComparableBitfieldFilled&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableBitfield));

union U {
  int i;

  friend bool operator==(const U&, const U&) = default;
};

struct NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion {
  U u;

  friend bool operator==(const NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion&, const NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableImplicitlyDeletedOperatorByUnion));

struct NotTriviallyEqualityComparableExplicitlyDeleted {
  int i;

  friend bool operator==(const NotTriviallyEqualityComparableExplicitlyDeleted&, const NotTriviallyEqualityComparableExplicitlyDeleted&) = delete;
};

struct NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct {
  NotTriviallyEqualityComparableExplicitlyDeleted u;

  friend bool operator==(const NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct&, const NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableImplicitlyDeletedOperatorByStruct));

struct NotTriviallyEqualityComparableHasReferenceMember {
  int& i;

  friend bool operator==(const NotTriviallyEqualityComparableHasReferenceMember&, const NotTriviallyEqualityComparableHasReferenceMember&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasReferenceMember));

enum E {
  a,
  b
};
bool operator==(E, E) { return false; }
static_assert(!__is_trivially_equality_comparable(E));

struct NotTriviallyEqualityComparableHasEnum {
  E e;
  friend bool operator==(const NotTriviallyEqualityComparableHasEnum&, const NotTriviallyEqualityComparableHasEnum&) = default;
};
static_assert(!__is_trivially_equality_comparable(NotTriviallyEqualityComparableHasEnum));

struct NonTriviallyEqualityComparableValueComparisonNonTriviallyCopyable {
  int i;
  NonTriviallyEqualityComparableValueComparisonNonTriviallyCopyable(const NonTriviallyEqualityComparableValueComparisonNonTriviallyCopyable&);

  friend bool operator==(NonTriviallyEqualityComparableValueComparisonNonTriviallyCopyable, NonTriviallyEqualityComparableValueComparisonNonTriviallyCopyable) = default;
};
static_assert(!__is_trivially_equality_comparable(NonTriviallyEqualityComparableValueComparisonNonTriviallyCopyable));

struct TriviallyEqualityComparableRefComparisonNonTriviallyCopyable {
  int i;
  TriviallyEqualityComparableRefComparisonNonTriviallyCopyable(const TriviallyEqualityComparableRefComparisonNonTriviallyCopyable&);

  friend bool operator==(const TriviallyEqualityComparableRefComparisonNonTriviallyCopyable&, const TriviallyEqualityComparableRefComparisonNonTriviallyCopyable&) = default;
};
static_assert(__is_trivially_equality_comparable(TriviallyEqualityComparableRefComparisonNonTriviallyCopyable));
}

#endif // __cplusplus >= 202002L
};

namespace can_pass_in_regs {

struct A { };

struct B {
  ~B();
};

struct C; // expected-note {{forward declaration}}

union D {
  int x;
};

static_assert(__can_pass_in_regs(A));
static_assert(__can_pass_in_regs(A));
static_assert(!__can_pass_in_regs(B));
static_assert(__can_pass_in_regs(D));

void test_errors() {
  (void)__can_pass_in_regs(const A); // expected-error {{not an unqualified class type}}
  (void)__can_pass_in_regs(A&); // expected-error {{not an unqualified class type}}
  (void)__can_pass_in_regs(A&&); // expected-error {{not an unqualified class type}}
  (void)__can_pass_in_regs(const A&); // expected-error {{not an unqualified class type}}
  (void)__can_pass_in_regs(C); // expected-error {{incomplete type}}
  (void)__can_pass_in_regs(int); // expected-error {{not an unqualified class type}}
  (void)__can_pass_in_regs(int&); // expected-error {{not an unqualified class type}}
}
}

struct S {};
template <class T> using remove_const_t = __remove_const(T);

void check_remove_const() {
  static_assert(__is_same(remove_const_t<void>, void));
  static_assert(__is_same(remove_const_t<const void>, void));
  static_assert(__is_same(remove_const_t<int>, int));
  static_assert(__is_same(remove_const_t<const int>, int));
  static_assert(__is_same(remove_const_t<volatile int>, volatile int));
  static_assert(__is_same(remove_const_t<const volatile int>, volatile int));
  static_assert(__is_same(remove_const_t<int *>, int *));
  static_assert(__is_same(remove_const_t<int *const>, int *));
  static_assert(__is_same(remove_const_t<int const *const>, int const *));
  static_assert(__is_same(remove_const_t<int const *const __restrict>, int const *__restrict));
  static_assert(__is_same(remove_const_t<int &>, int &));
  static_assert(__is_same(remove_const_t<int const &>, int const &));
  static_assert(__is_same(remove_const_t<int &&>, int &&));
  static_assert(__is_same(remove_const_t<int const &&>, int const &&));
  static_assert(__is_same(remove_const_t<int()>, int()));
  static_assert(__is_same(remove_const_t<int (*const)()>, int (*)()));
  static_assert(__is_same(remove_const_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_const_t<S>, S));
  static_assert(__is_same(remove_const_t<const S>, S));
  static_assert(__is_same(remove_const_t<volatile S>, volatile S));
  static_assert(__is_same(remove_const_t<S *__restrict>, S *__restrict));
  static_assert(__is_same(remove_const_t<const volatile S>, volatile S));
  static_assert(__is_same(remove_const_t<S *const volatile __restrict>, S *volatile __restrict));
  static_assert(__is_same(remove_const_t<int S::*const>, int S::*));
  static_assert(__is_same(remove_const_t<int (S::*const)()>, int(S::*)()));
}

template <class T> using remove_restrict_t = __remove_restrict(T);

void check_remove_restrict() {
  static_assert(__is_same(remove_restrict_t<void>, void));
  static_assert(__is_same(remove_restrict_t<int>, int));
  static_assert(__is_same(remove_restrict_t<const int>, const int));
  static_assert(__is_same(remove_restrict_t<volatile int>, volatile int));
  static_assert(__is_same(remove_restrict_t<int *__restrict>, int *));
  static_assert(__is_same(remove_restrict_t<int *const volatile __restrict>, int *const volatile));
  static_assert(__is_same(remove_restrict_t<int *>, int *));
  static_assert(__is_same(remove_restrict_t<int *__restrict>, int *));
  static_assert(__is_same(remove_restrict_t<int &>, int &));
  static_assert(__is_same(remove_restrict_t<int &__restrict>, int &));
  static_assert(__is_same(remove_restrict_t<int &&>, int &&));
  static_assert(__is_same(remove_restrict_t<int &&__restrict>, int &&));
  static_assert(__is_same(remove_restrict_t<int()>, int()));
  static_assert(__is_same(remove_restrict_t<int (*const volatile)()>, int (*const volatile)()));
  static_assert(__is_same(remove_restrict_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_restrict_t<S>, S));
  static_assert(__is_same(remove_restrict_t<const S>, const S));
  static_assert(__is_same(remove_restrict_t<volatile S>, volatile S));
  static_assert(__is_same(remove_restrict_t<S *__restrict>, S *));
  static_assert(__is_same(remove_restrict_t<S *const volatile __restrict>, S *const volatile));
  static_assert(__is_same(remove_restrict_t<int S::*__restrict>, int S::*));
  static_assert(__is_same(remove_restrict_t<int (S::*const volatile)()>, int(S::*const volatile)()));
}

template <class T> using remove_volatile_t = __remove_volatile(T);

void check_remove_volatile() {
  static_assert(__is_same(remove_volatile_t<void>, void));
  static_assert(__is_same(remove_volatile_t<volatile void>, void));
  static_assert(__is_same(remove_volatile_t<int>, int));
  static_assert(__is_same(remove_volatile_t<const int>, const int));
  static_assert(__is_same(remove_volatile_t<volatile int>, int));
  static_assert(__is_same(remove_volatile_t<int *__restrict>, int *__restrict));
  static_assert(__is_same(remove_volatile_t<const volatile int>, const int));
  static_assert(__is_same(remove_volatile_t<int *const volatile __restrict>, int *const __restrict));
  static_assert(__is_same(remove_volatile_t<int *>, int *));
  static_assert(__is_same(remove_volatile_t<int *volatile>, int *));
  static_assert(__is_same(remove_volatile_t<int volatile *volatile>, int volatile *));
  static_assert(__is_same(remove_volatile_t<int &>, int &));
  static_assert(__is_same(remove_volatile_t<int volatile &>, int volatile &));
  static_assert(__is_same(remove_volatile_t<int &&>, int &&));
  static_assert(__is_same(remove_volatile_t<int volatile &&>, int volatile &&));
  static_assert(__is_same(remove_volatile_t<int()>, int()));
  static_assert(__is_same(remove_volatile_t<int (*volatile)()>, int (*)()));
  static_assert(__is_same(remove_volatile_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_volatile_t<S>, S));
  static_assert(__is_same(remove_volatile_t<const S>, const S));
  static_assert(__is_same(remove_volatile_t<volatile S>, S));
  static_assert(__is_same(remove_volatile_t<const volatile S>, const S));
  static_assert(__is_same(remove_volatile_t<int S::*volatile>, int S::*));
  static_assert(__is_same(remove_volatile_t<int (S::*volatile)()>, int(S::*)()));
}

template <class T> using remove_cv_t = __remove_cv(T);

void check_remove_cv() {
  static_assert(__is_same(remove_cv_t<void>, void));
  static_assert(__is_same(remove_cv_t<const volatile void>, void));
  static_assert(__is_same(remove_cv_t<int>, int));
  static_assert(__is_same(remove_cv_t<const int>, int));
  static_assert(__is_same(remove_cv_t<volatile int>, int));
  static_assert(__is_same(remove_cv_t<const volatile int>, int));
  static_assert(__is_same(remove_cv_t<int *>, int *));
  static_assert(__is_same(remove_cv_t<int *const volatile>, int *));
  static_assert(__is_same(remove_cv_t<int const *const volatile>, int const *));
  static_assert(__is_same(remove_cv_t<int const *const volatile __restrict>, int const *__restrict));
  static_assert(__is_same(remove_cv_t<int const *const volatile _Nonnull>, int const *_Nonnull));
  static_assert(__is_same(remove_cv_t<int &>, int &));
  static_assert(__is_same(remove_cv_t<int const volatile &>, int const volatile &));
  static_assert(__is_same(remove_cv_t<int &&>, int &&));
  static_assert(__is_same(remove_cv_t<int const volatile &&>, int const volatile &&));
  static_assert(__is_same(remove_cv_t<int()>, int()));
  static_assert(__is_same(remove_cv_t<int (*const volatile)()>, int (*)()));
  static_assert(__is_same(remove_cv_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_cv_t<S>, S));
  static_assert(__is_same(remove_cv_t<const S>, S));
  static_assert(__is_same(remove_cv_t<volatile S>, S));
  static_assert(__is_same(remove_cv_t<const volatile S>, S));
  static_assert(__is_same(remove_cv_t<int S::*const volatile>, int S::*));
  static_assert(__is_same(remove_cv_t<int (S::*const volatile)()>, int(S::*)()));
}

template <class T> using add_pointer_t = __add_pointer(T);

void add_pointer() {
  static_assert(__is_same(add_pointer_t<void>, void *));
  static_assert(__is_same(add_pointer_t<const void>, const void *));
  static_assert(__is_same(add_pointer_t<volatile void>, volatile void *));
  static_assert(__is_same(add_pointer_t<const volatile void>, const volatile void *));
  static_assert(__is_same(add_pointer_t<int>, int *));
  static_assert(__is_same(add_pointer_t<const int>, const int *));
  static_assert(__is_same(add_pointer_t<volatile int>, volatile int *));
  static_assert(__is_same(add_pointer_t<const volatile int>, const volatile int *));
  static_assert(__is_same(add_pointer_t<int *>, int **));
  static_assert(__is_same(add_pointer_t<int &>, int *));
  static_assert(__is_same(add_pointer_t<int &&>, int *));
  static_assert(__is_same(add_pointer_t<int()>, int (*)()));
  static_assert(__is_same(add_pointer_t<int (*)()>, int (**)()));
  static_assert(__is_same(add_pointer_t<int (&)()>, int (*)()));

  static_assert(__is_same(add_pointer_t<S>, S *));
  static_assert(__is_same(add_pointer_t<const S>, const S *));
  static_assert(__is_same(add_pointer_t<volatile S>, volatile S *));
  static_assert(__is_same(add_pointer_t<const volatile S>, const volatile S *));
  static_assert(__is_same(add_pointer_t<int S::*>, int S::**));
  static_assert(__is_same(add_pointer_t<int (S::*)()>, int(S::**)()));

  static_assert(__is_same(add_pointer_t<int __attribute__((address_space(1)))>, int __attribute__((address_space(1))) *));
  static_assert(__is_same(add_pointer_t<S __attribute__((address_space(2)))>, S __attribute__((address_space(2))) *));
}

template <class T> using remove_pointer_t = __remove_pointer(T);

void remove_pointer() {
  static_assert(__is_same(remove_pointer_t<void>, void));
  static_assert(__is_same(remove_pointer_t<const void>, const void));
  static_assert(__is_same(remove_pointer_t<volatile void>, volatile void));
  static_assert(__is_same(remove_pointer_t<const volatile void>, const volatile void));
  static_assert(__is_same(remove_pointer_t<int>, int));
  static_assert(__is_same(remove_pointer_t<const int>, const int));
  static_assert(__is_same(remove_pointer_t<volatile int>, volatile int));
  static_assert(__is_same(remove_pointer_t<const volatile int>, const volatile int));
  static_assert(__is_same(remove_pointer_t<int *>, int));
  static_assert(__is_same(remove_pointer_t<const int *>, const int));
  static_assert(__is_same(remove_pointer_t<volatile int *>, volatile int));
  static_assert(__is_same(remove_pointer_t<const volatile int *>, const volatile int));
  static_assert(__is_same(remove_pointer_t<int *const>, int));
  static_assert(__is_same(remove_pointer_t<int *volatile>, int));
  static_assert(__is_same(remove_pointer_t<int *const volatile>, int));
  static_assert(__is_same(remove_pointer_t<int &>, int &));
  static_assert(__is_same(remove_pointer_t<int &&>, int &&));
  static_assert(__is_same(remove_pointer_t<int()>, int()));
  static_assert(__is_same(remove_pointer_t<int (*)()>, int()));
  static_assert(__is_same(remove_pointer_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_pointer_t<S>, S));
  static_assert(__is_same(remove_pointer_t<const S>, const S));
  static_assert(__is_same(remove_pointer_t<volatile S>, volatile S));
  static_assert(__is_same(remove_pointer_t<const volatile S>, const volatile S));
  static_assert(__is_same(remove_pointer_t<int S::*>, int S::*));
  static_assert(__is_same(remove_pointer_t<int (S::*)()>, int(S::*)()));

  static_assert(__is_same(remove_pointer_t<int __attribute__((address_space(1))) *>, int __attribute__((address_space(1)))));
  static_assert(__is_same(remove_pointer_t<S __attribute__((address_space(2))) *>, S  __attribute__((address_space(2)))));

  static_assert(__is_same(remove_pointer_t<int (^)(char)>, int (^)(char)));
}

template <class T> using add_lvalue_reference_t = __add_lvalue_reference(T);

void add_lvalue_reference() {
  static_assert(__is_same(add_lvalue_reference_t<void>, void));
  static_assert(__is_same(add_lvalue_reference_t<const void>, const void));
  static_assert(__is_same(add_lvalue_reference_t<volatile void>, volatile void));
  static_assert(__is_same(add_lvalue_reference_t<const volatile void>, const volatile void));
  static_assert(__is_same(add_lvalue_reference_t<int>, int &));
  static_assert(__is_same(add_lvalue_reference_t<const int>, const int &));
  static_assert(__is_same(add_lvalue_reference_t<volatile int>, volatile int &));
  static_assert(__is_same(add_lvalue_reference_t<const volatile int>, const volatile int &));
  static_assert(__is_same(add_lvalue_reference_t<int *>, int *&));
  static_assert(__is_same(add_lvalue_reference_t<int &>, int &));
  static_assert(__is_same(add_lvalue_reference_t<int &&>, int &)); // reference collapsing
  static_assert(__is_same(add_lvalue_reference_t<int()>, int (&)()));
  static_assert(__is_same(add_lvalue_reference_t<int (*)()>, int (*&)()));
  static_assert(__is_same(add_lvalue_reference_t<int (&)()>, int (&)()));

  static_assert(__is_same(add_lvalue_reference_t<S>, S &));
  static_assert(__is_same(add_lvalue_reference_t<const S>, const S &));
  static_assert(__is_same(add_lvalue_reference_t<volatile S>, volatile S &));
  static_assert(__is_same(add_lvalue_reference_t<const volatile S>, const volatile S &));
  static_assert(__is_same(add_lvalue_reference_t<int S::*>, int S::*&));
  static_assert(__is_same(add_lvalue_reference_t<int (S::*)()>, int(S::*&)()));
}

template <class T> using add_rvalue_reference_t = __add_rvalue_reference(T);

void add_rvalue_reference() {
  static_assert(__is_same(add_rvalue_reference_t<void>, void));
  static_assert(__is_same(add_rvalue_reference_t<const void>, const void));
  static_assert(__is_same(add_rvalue_reference_t<volatile void>, volatile void));
  static_assert(__is_same(add_rvalue_reference_t<const volatile void>, const volatile void));
  static_assert(__is_same(add_rvalue_reference_t<int>, int &&));
  static_assert(__is_same(add_rvalue_reference_t<const int>, const int &&));
  static_assert(__is_same(add_rvalue_reference_t<volatile int>, volatile int &&));
  static_assert(__is_same(add_rvalue_reference_t<const volatile int>, const volatile int &&));
  static_assert(__is_same(add_rvalue_reference_t<int *>, int *&&));
  static_assert(__is_same(add_rvalue_reference_t<int &>, int &)); // reference collapsing
  static_assert(__is_same(add_rvalue_reference_t<int &&>, int &&));
  static_assert(__is_same(add_rvalue_reference_t<int()>, int(&&)()));
  static_assert(__is_same(add_rvalue_reference_t<int (*)()>, int (*&&)()));
  static_assert(__is_same(add_rvalue_reference_t<int (&)()>, int (&)())); // reference collapsing

  static_assert(__is_same(add_rvalue_reference_t<S>, S &&));
  static_assert(__is_same(add_rvalue_reference_t<const S>, const S &&));
  static_assert(__is_same(add_rvalue_reference_t<volatile S>, volatile S &&));
  static_assert(__is_same(add_rvalue_reference_t<const volatile S>, const volatile S &&));
  static_assert(__is_same(add_rvalue_reference_t<int S::*>, int S::*&&));
  static_assert(__is_same(add_rvalue_reference_t<int (S::*)()>, int(S::* &&)()));
}

template <class T> using remove_reference_t = __remove_reference_t(T);

void check_remove_reference() {
  static_assert(__is_same(remove_reference_t<void>, void));
  static_assert(__is_same(remove_reference_t<const volatile void>, const volatile void));
  static_assert(__is_same(remove_reference_t<int>, int));
  static_assert(__is_same(remove_reference_t<const int>, const int));
  static_assert(__is_same(remove_reference_t<volatile int>, volatile int));
  static_assert(__is_same(remove_reference_t<const volatile int>, const volatile int));
  static_assert(__is_same(remove_reference_t<int *>, int *));
  static_assert(__is_same(remove_reference_t<int *const volatile>, int *const volatile));
  static_assert(__is_same(remove_reference_t<int const *const volatile>, int const *const volatile));
  static_assert(__is_same(remove_reference_t<int &>, int));
  static_assert(__is_same(remove_reference_t<int const volatile &>, int const volatile));
  static_assert(__is_same(remove_reference_t<int &&>, int));
  static_assert(__is_same(remove_reference_t<int const volatile &&>, int const volatile));
  static_assert(__is_same(remove_reference_t<int()>, int()));
  static_assert(__is_same(remove_reference_t<int (*const volatile)()>, int (*const volatile)()));
  static_assert(__is_same(remove_reference_t<int (&)()>, int()));

  static_assert(__is_same(remove_reference_t<S>, S));
  static_assert(__is_same(remove_reference_t<S &>, S));
  static_assert(__is_same(remove_reference_t<S &&>, S));
  static_assert(__is_same(remove_reference_t<const S>, const S));
  static_assert(__is_same(remove_reference_t<const S &>, const S));
  static_assert(__is_same(remove_reference_t<const S &&>, const S));
  static_assert(__is_same(remove_reference_t<volatile S>, volatile S));
  static_assert(__is_same(remove_reference_t<volatile S &>, volatile S));
  static_assert(__is_same(remove_reference_t<volatile S &&>, volatile S));
  static_assert(__is_same(remove_reference_t<const volatile S>, const volatile S));
  static_assert(__is_same(remove_reference_t<const volatile S &>, const volatile S));
  static_assert(__is_same(remove_reference_t<const volatile S &&>, const volatile S));
  static_assert(__is_same(remove_reference_t<int S::*const volatile &>, int S::*const volatile));
  static_assert(__is_same(remove_reference_t<int (S::*const volatile &)()>, int(S::*const volatile)()));
  static_assert(__is_same(remove_reference_t<int (S::*const volatile &&)() &>, int(S::*const volatile)() &));
}

template <class T> using remove_cvref_t = __remove_cvref(T);

void check_remove_cvref() {
  static_assert(__is_same(remove_cvref_t<void>, void));
  static_assert(__is_same(remove_cvref_t<const volatile void>, void));
  static_assert(__is_same(remove_cvref_t<int>, int));
  static_assert(__is_same(remove_cvref_t<const int>, int));
  static_assert(__is_same(remove_cvref_t<volatile int>, int));
  static_assert(__is_same(remove_cvref_t<const volatile int>, int));
  static_assert(__is_same(remove_cvref_t<int *>, int *));
  static_assert(__is_same(remove_cvref_t<int *const volatile>, int *));
  static_assert(__is_same(remove_cvref_t<int const *const volatile>, int const *));
  static_assert(__is_same(remove_cvref_t<int const *const volatile __restrict>, int const *__restrict));
  static_assert(__is_same(remove_cvref_t<int const *const volatile _Nonnull>, int const *_Nonnull));
  static_assert(__is_same(remove_cvref_t<int &>, int));
  static_assert(__is_same(remove_cvref_t<int const volatile &>, int));
  static_assert(__is_same(remove_cvref_t<int &&>, int));
  static_assert(__is_same(remove_cvref_t<int const volatile &&>, int));
  static_assert(__is_same(remove_cvref_t<int()>, int()));
  static_assert(__is_same(remove_cvref_t<int (*const volatile)()>, int (*)()));
  static_assert(__is_same(remove_cvref_t<int (&)()>, int()));

  static_assert(__is_same(remove_cvref_t<S>, S));
  static_assert(__is_same(remove_cvref_t<S &>, S));
  static_assert(__is_same(remove_cvref_t<S &&>, S));
  static_assert(__is_same(remove_cvref_t<const S>, S));
  static_assert(__is_same(remove_cvref_t<const S &>, S));
  static_assert(__is_same(remove_cvref_t<const S &&>, S));
  static_assert(__is_same(remove_cvref_t<volatile S>, S));
  static_assert(__is_same(remove_cvref_t<volatile S &>, S));
  static_assert(__is_same(remove_cvref_t<volatile S &&>, S));
  static_assert(__is_same(remove_cvref_t<const volatile S>, S));
  static_assert(__is_same(remove_cvref_t<const volatile S &>, S));
  static_assert(__is_same(remove_cvref_t<const volatile S &&>, S));
  static_assert(__is_same(remove_cvref_t<int S::*const volatile>, int S::*));
  static_assert(__is_same(remove_cvref_t<int (S::*const volatile)()>, int(S::*)()));
  static_assert(__is_same(remove_cvref_t<int (S::*const volatile)() &>, int(S::*)() &));
  static_assert(__is_same(remove_cvref_t<int (S::*const volatile)() &&>, int(S::*)() &&));
}

template <class T> using decay_t = __decay(T);

void check_decay() {
  static_assert(__is_same(decay_t<void>, void));
  static_assert(__is_same(decay_t<const volatile void>, void));
  static_assert(__is_same(decay_t<int>, int));
  static_assert(__is_same(decay_t<const int>, int));
  static_assert(__is_same(decay_t<volatile int>, int));
  static_assert(__is_same(decay_t<const volatile int>, int));
  static_assert(__is_same(decay_t<int *>, int *));
  static_assert(__is_same(decay_t<int *const volatile>, int *));
  static_assert(__is_same(decay_t<int *const volatile __restrict>, int *));
  static_assert(__is_same(decay_t<int const *const volatile>, int const *));
  static_assert(__is_same(decay_t<int const *const volatile _Nonnull>, int const *));
  static_assert(__is_same(decay_t<int &>, int));
  static_assert(__is_same(decay_t<int const volatile &>, int));
  static_assert(__is_same(decay_t<int &&>, int));
  static_assert(__is_same(decay_t<int const volatile &&>, int));
  static_assert(__is_same(decay_t<int()>, int (*)()));
  static_assert(__is_same(decay_t<int (*)()>, int (*)()));
  static_assert(__is_same(decay_t<int (*const)()>, int (*)()));
  static_assert(__is_same(decay_t<int (*volatile)()>, int (*)()));
  static_assert(__is_same(decay_t<int (*const volatile)()>, int (*)()));
  static_assert(__is_same(decay_t<int (&)()>, int (*)()));
  static_assert(__is_same(decay_t<IntAr>, int *));
  static_assert(__is_same(decay_t<IntArNB>, int *));

  static_assert(__is_same(decay_t<S>, S));
  static_assert(__is_same(decay_t<S &>, S));
  static_assert(__is_same(decay_t<S &&>, S));
  static_assert(__is_same(decay_t<const S>, S));
  static_assert(__is_same(decay_t<const S &>, S));
  static_assert(__is_same(decay_t<const S &&>, S));
  static_assert(__is_same(decay_t<volatile S>, S));
  static_assert(__is_same(decay_t<volatile S &>, S));
  static_assert(__is_same(decay_t<volatile S &&>, S));
  static_assert(__is_same(decay_t<const volatile S>, S));
  static_assert(__is_same(decay_t<const volatile S &>, S));
  static_assert(__is_same(decay_t<const volatile S &&>, S));
  static_assert(__is_same(decay_t<int S::*const volatile>, int S::*));
  static_assert(__is_same(decay_t<int (S::*const volatile)()>, int(S::*)()));
  static_assert(__is_same(decay_t<int S::*const volatile &>, int S::*));
  static_assert(__is_same(decay_t<int (S::*const volatile &)()>, int(S::*)()));
  static_assert(__is_same(decay_t<int S::*const volatile &&>, int S::*));
}

template <class T> struct CheckAbominableFunction {};
template <class M>
struct CheckAbominableFunction<M S::*> {
  static void checks() {
    static_assert(__is_same(add_lvalue_reference_t<M>, M));
    static_assert(__is_same(add_pointer_t<M>, M));
    static_assert(__is_same(add_rvalue_reference_t<M>, M));
    static_assert(__is_same(decay_t<M>, M));
    static_assert(__is_same(remove_const_t<M>, M));
    static_assert(__is_same(remove_volatile_t<M>, M));
    static_assert(__is_same(remove_cv_t<M>, M));
    static_assert(__is_same(remove_cvref_t<M>, M));
    static_assert(__is_same(remove_pointer_t<M>, M));
    static_assert(__is_same(remove_reference_t<M>, M));
  }
};

void check_abominable_function() {
  { CheckAbominableFunction<int (S::*)() &> x; }
  { CheckAbominableFunction<int (S::*)() &&> x; }
  { CheckAbominableFunction<int (S::*)() const> x; }
  { CheckAbominableFunction<int (S::*)() const &> x; }
  { CheckAbominableFunction<int (S::*)() const &&> x; }
  { CheckAbominableFunction<int (S::*)() volatile> x; }
  { CheckAbominableFunction<int (S::*)() volatile &> x; }
  { CheckAbominableFunction<int (S::*)() volatile &&> x; }
  { CheckAbominableFunction<int (S::*)() const volatile> x; }
  { CheckAbominableFunction<int (S::*)() const volatile &> x; }
  { CheckAbominableFunction<int (S::*)() const volatile &&> x; }
}

template <class T> using make_signed_t = __make_signed(T);
template <class T, class Expected>
void check_make_signed() {
  static_assert(__is_same(make_signed_t<T>, Expected));
  static_assert(__is_same(make_signed_t<const T>, const Expected));
  static_assert(__is_same(make_signed_t<volatile T>, volatile Expected));
  static_assert(__is_same(make_signed_t<const volatile T>, const volatile Expected));
}

#if defined(__ILP32__) || defined(__LLP64__)
  using Int64 = long long;
  using UInt64 = unsigned long long;
#elif defined(__LP64__) || defined(__ILP64__) || defined(__SILP64__)
  using Int64 = long;
  using UInt64 = unsigned long;
#else
#error Programming model currently unsupported; please add a new entry.
#endif

enum UnscopedBool : bool {}; // deliberately char
enum class ScopedBool : bool {}; // deliberately char
enum UnscopedChar : char {}; // deliberately char
enum class ScopedChar : char {}; // deliberately char
enum UnscopedUChar : unsigned char {};
enum class ScopedUChar : unsigned char {};
enum UnscopedLongLong : long long {};
enum UnscopedULongLong : unsigned long long {};
enum class ScopedLongLong : long long {};
enum class ScopedULongLong : unsigned long long {};
enum class UnscopedInt128 : __int128 {};
enum class ScopedInt128 : __int128 {};
enum class UnscopedUInt128 : unsigned __int128 {};
enum class ScopedUInt128 : unsigned __int128 {};

void make_signed() {
  check_make_signed<char, signed char>();
  check_make_signed<signed char, signed char>();
  check_make_signed<unsigned char, signed char>();
  check_make_signed<short, short>();
  check_make_signed<unsigned short, short>();
  check_make_signed<int, int>();
  check_make_signed<unsigned int, int>();
  check_make_signed<long, long>();
  check_make_signed<unsigned long, long>();
  check_make_signed<long long, long long>();
  check_make_signed<unsigned long long, long long>();
  check_make_signed<__int128, __int128>();
  check_make_signed<__uint128_t, __int128>();
  check_make_signed<_BitInt(65), _BitInt(65)>();
  check_make_signed<unsigned _BitInt(65), _BitInt(65)>();

  check_make_signed<wchar_t, int>();
#if __cplusplus >= 202002L
  check_make_signed<char8_t, signed char>();
#endif
#if __cplusplus >= 201103L
  check_make_signed<char16_t, short>();
  check_make_signed<char32_t, int>();
#endif

  check_make_signed<UnscopedChar, signed char>();
  check_make_signed<ScopedChar, signed char>();
  check_make_signed<UnscopedUChar, signed char>();
  check_make_signed<ScopedUChar, signed char>();

  check_make_signed<UnscopedLongLong, Int64>();
  check_make_signed<UnscopedULongLong, Int64>();
  check_make_signed<ScopedLongLong, Int64>();
  check_make_signed<ScopedULongLong, Int64>();

  check_make_signed<UnscopedInt128, __int128>();
  check_make_signed<ScopedInt128, __int128>();
  check_make_signed<UnscopedUInt128, __int128>();
  check_make_signed<ScopedUInt128, __int128>();

  { using ExpectedError = __make_signed(bool); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'bool'}}
  { using ExpectedError = __make_signed(UnscopedBool); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'UnscopedBool' whose underlying type is 'bool'}}
  { using ExpectedError = __make_signed(ScopedBool); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'ScopedBool' whose underlying type is 'bool'}}
  { using ExpectedError = __make_signed(unsigned _BitInt(1)); }
  // expected-error@*:*{{'make_signed' is only compatible with non-_BitInt(1) integers and enum types, but was given 'unsigned _BitInt(1)'}}
  { using ExpectedError = __make_signed(int[]); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int[]'}}
  { using ExpectedError = __make_signed(int[5]); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int[5]'}}
  { using ExpectedError = __make_signed(void); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'void'}}
  { using ExpectedError = __make_signed(int *); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int *'}}
  { using ExpectedError = __make_signed(int &); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int &'}}
  { using ExpectedError = __make_signed(int &&); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int &&'}}
  { using ExpectedError = __make_signed(float); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'float'}}
  { using ExpectedError = __make_signed(double); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'double'}}
  { using ExpectedError = __make_signed(long double); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'long double'}}
  { using ExpectedError = __make_signed(S); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'S'}}
  { using ExpectedError = __make_signed(S *); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'S *'}}
  { using ExpectedError = __make_signed(int S::*); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int S::*'}}
  { using ExpectedError = __make_signed(int(S::*)()); }
  // expected-error@*:*{{'make_signed' is only compatible with non-bool integers and enum types, but was given 'int (S::*)()'}}
}

template <class T>
using make_unsigned_t = __make_unsigned(T);

template <class T, class Expected>
void check_make_unsigned() {
  static_assert(__is_same(make_unsigned_t<T>, Expected));
  static_assert(__is_same(make_unsigned_t<const T>, const Expected));
  static_assert(__is_same(make_unsigned_t<volatile T>, volatile Expected));
  static_assert(__is_same(make_unsigned_t<const volatile T>, const volatile Expected));
}

void make_unsigned() {
  check_make_unsigned<char, unsigned char>();
  check_make_unsigned<signed char, unsigned char>();
  check_make_unsigned<unsigned char, unsigned char>();
  check_make_unsigned<short, unsigned short>();
  check_make_unsigned<unsigned short, unsigned short>();
  check_make_unsigned<int, unsigned int>();
  check_make_unsigned<unsigned int, unsigned int>();
  check_make_unsigned<long, unsigned long>();
  check_make_unsigned<unsigned long, unsigned long>();
  check_make_unsigned<long long, unsigned long long>();
  check_make_unsigned<unsigned long long, unsigned long long>();
  check_make_unsigned<__int128, __uint128_t>();
  check_make_unsigned<__uint128_t, __uint128_t>();
  check_make_unsigned<_BitInt(65), unsigned _BitInt(65)>();
  check_make_unsigned<unsigned _BitInt(65), unsigned _BitInt(65)>();

  check_make_unsigned<wchar_t, unsigned int>();
#if __cplusplus >= 202002L
  check_make_unsigned<char8_t, unsigned char>();
#endif
#if __cplusplus >= 201103L
  check_make_unsigned<char16_t, unsigned short>();
  check_make_unsigned<char32_t, unsigned int>();
#endif

  check_make_unsigned<UnscopedChar, unsigned char>();
  check_make_unsigned<ScopedChar, unsigned char>();
  check_make_unsigned<UnscopedUChar, unsigned char>();
  check_make_unsigned<ScopedUChar, unsigned char>();

  check_make_unsigned<UnscopedLongLong, UInt64>();
  check_make_unsigned<UnscopedULongLong, UInt64>();
  check_make_unsigned<ScopedLongLong, UInt64>();
  check_make_unsigned<ScopedULongLong, UInt64>();

  check_make_unsigned<UnscopedInt128, unsigned __int128>();
  check_make_unsigned<ScopedInt128, unsigned __int128>();
  check_make_unsigned<UnscopedUInt128, unsigned __int128>();
  check_make_unsigned<ScopedUInt128, unsigned __int128>();

  { using ExpectedError = __make_unsigned(bool); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'bool'}}
  { using ExpectedError = __make_unsigned(UnscopedBool); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'UnscopedBool' whose underlying type is 'bool'}}
  { using ExpectedError = __make_unsigned(ScopedBool); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'ScopedBool' whose underlying type is 'bool'}}
  { using ExpectedError = __make_unsigned(unsigned _BitInt(1)); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-_BitInt(1) integers and enum types, but was given 'unsigned _BitInt(1)'}}
  { using ExpectedError = __make_unsigned(int[]); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int[]'}}
  { using ExpectedError = __make_unsigned(int[5]); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int[5]'}}
  { using ExpectedError = __make_unsigned(void); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'void'}}
  { using ExpectedError = __make_unsigned(int *); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int *'}}
  { using ExpectedError = __make_unsigned(int &); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int &'}}
  { using ExpectedError = __make_unsigned(int &&); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int &&'}}
  { using ExpectedError = __make_unsigned(float); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'float'}}
  { using ExpectedError = __make_unsigned(double); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'double'}}
  { using ExpectedError = __make_unsigned(long double); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'long double'}}
  { using ExpectedError = __make_unsigned(S); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'S'}}
  { using ExpectedError = __make_unsigned(S *); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'S *'}}
  { using ExpectedError = __make_unsigned(int S::*); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int S::*'}}
  { using ExpectedError = __make_unsigned(int(S::*)()); }
  // expected-error@*:*{{'make_unsigned' is only compatible with non-bool integers and enum types, but was given 'int (S::*)()'}}
}

template <class T> using remove_extent_t = __remove_extent(T);

void remove_extent() {
  static_assert(__is_same(remove_extent_t<void>, void));
  static_assert(__is_same(remove_extent_t<int>, int));
  static_assert(__is_same(remove_extent_t<int[]>, int));
  static_assert(__is_same(remove_extent_t<int[1]>, int));
  static_assert(__is_same(remove_extent_t<int[1][2]>, int[2]));
  static_assert(__is_same(remove_extent_t<int[][2]>, int[2]));
  static_assert(__is_same(remove_extent_t<const int[]>, const int));
  static_assert(__is_same(remove_extent_t<const int[1]>, const int));
  static_assert(__is_same(remove_extent_t<const int[1][2]>, const int[2]));
  static_assert(__is_same(remove_extent_t<const int[][2]>, const int[2]));
  static_assert(__is_same(remove_extent_t<volatile int[]>, volatile int));
  static_assert(__is_same(remove_extent_t<volatile int[1]>, volatile int));
  static_assert(__is_same(remove_extent_t<volatile int[1][2]>, volatile int[2]));
  static_assert(__is_same(remove_extent_t<volatile int[][2]>, volatile int[2]));
  static_assert(__is_same(remove_extent_t<const volatile int[]>, const volatile int));
  static_assert(__is_same(remove_extent_t<const volatile int[1]>, const volatile int));
  static_assert(__is_same(remove_extent_t<const volatile int[1][2]>, const volatile int[2]));
  static_assert(__is_same(remove_extent_t<const volatile int[][2]>, const volatile int[2]));
  static_assert(__is_same(remove_extent_t<int *>, int *));
  static_assert(__is_same(remove_extent_t<int &>, int &));
  static_assert(__is_same(remove_extent_t<int &&>, int &&));
  static_assert(__is_same(remove_extent_t<int()>, int()));
  static_assert(__is_same(remove_extent_t<int (*)()>, int (*)()));
  static_assert(__is_same(remove_extent_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_extent_t<S>, S));
  static_assert(__is_same(remove_extent_t<int S::*>, int S::*));
  static_assert(__is_same(remove_extent_t<int (S::*)()>, int(S::*)()));

  using SomeArray = int[1][2];
  static_assert(__is_same(remove_extent_t<const SomeArray>, const int[2]));
}

template <class T> using remove_all_extents_t = __remove_all_extents(T);

void remove_all_extents() {
  static_assert(__is_same(remove_all_extents_t<void>, void));
  static_assert(__is_same(remove_all_extents_t<int>, int));
  static_assert(__is_same(remove_all_extents_t<const int>, const int));
  static_assert(__is_same(remove_all_extents_t<volatile int>, volatile int));
  static_assert(__is_same(remove_all_extents_t<const volatile int>, const volatile int));
  static_assert(__is_same(remove_all_extents_t<int[]>, int));
  static_assert(__is_same(remove_all_extents_t<int[1]>, int));
  static_assert(__is_same(remove_all_extents_t<int[1][2]>, int));
  static_assert(__is_same(remove_all_extents_t<int[][2]>, int));
  static_assert(__is_same(remove_all_extents_t<const int[]>, const int));
  static_assert(__is_same(remove_all_extents_t<const int[1]>, const int));
  static_assert(__is_same(remove_all_extents_t<const int[1][2]>, const int));
  static_assert(__is_same(remove_all_extents_t<const int[][2]>, const int));
  static_assert(__is_same(remove_all_extents_t<volatile int[]>, volatile int));
  static_assert(__is_same(remove_all_extents_t<volatile int[1]>, volatile int));
  static_assert(__is_same(remove_all_extents_t<volatile int[1][2]>, volatile int));
  static_assert(__is_same(remove_all_extents_t<volatile int[][2]>, volatile int));
  static_assert(__is_same(remove_all_extents_t<const volatile int[]>, const volatile int));
  static_assert(__is_same(remove_all_extents_t<const volatile int[1]>, const volatile int));
  static_assert(__is_same(remove_all_extents_t<const volatile int[1][2]>, const volatile int));
  static_assert(__is_same(remove_all_extents_t<const volatile int[][2]>, const volatile int));
  static_assert(__is_same(remove_all_extents_t<int *>, int *));
  static_assert(__is_same(remove_all_extents_t<int &>, int &));
  static_assert(__is_same(remove_all_extents_t<int &&>, int &&));
  static_assert(__is_same(remove_all_extents_t<int()>, int()));
  static_assert(__is_same(remove_all_extents_t<int (*)()>, int (*)()));
  static_assert(__is_same(remove_all_extents_t<int (&)()>, int (&)()));

  static_assert(__is_same(remove_all_extents_t<S>, S));
  static_assert(__is_same(remove_all_extents_t<int S::*>, int S::*));
  static_assert(__is_same(remove_all_extents_t<int (S::*)()>, int(S::*)()));

  using SomeArray = int[1][2];
  static_assert(__is_same(remove_all_extents_t<const SomeArray>, const int));
}

namespace GH121278 {
// https://cplusplus.github.io/LWG/lwg-active.html#3929
#if __cplusplus >= 202002L
template <typename B, typename D>
concept C = __is_base_of(B, D);
// expected-error@-1 {{incomplete type 'S' used in type trait expression}}
// expected-note@-2 {{while substituting template arguments into constraint expression here}}

struct T;
struct S;
bool b = C<T, S>;
// expected-note@-1 {{while checking the satisfaction of concept 'C<T, S>' requested here}}
#endif
}
