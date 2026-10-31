// RUN: %clang_analyze_cc1 -std=c++20 %s \
// RUN:   -analyzer-checker=core,alpha.cplusplus.PlacementSelfCopy \
// RUN:   -analyzer-output=text -verify -triple x86_64-unknown-linux-gnu
// RUN: %clang_analyze_cc1 -std=c++20 %s -DDECLARED_PLACEMENT_NEW \
// RUN:   -analyzer-checker=core,alpha.cplusplus.PlacementSelfCopy \
// RUN:   -verify=declared -triple x86_64-unknown-linux-gnu

#ifdef DECLARED_PLACEMENT_NEW

// The reserved placement operator has no body, so its return value is not the
// placement argument.
void *operator new(__SIZE_TYPE__, void *P) noexcept;

struct Swapped {
  int X, Y;
  Swapped(const Swapped &O) : X(O.Y), Y(O.X) {} // declared-warning{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void swapped_copy(Swapped *Dst) {
  ::new (Dst) Swapped(*Dst);
}

#else

#include "Inputs/system-header-simulator-cxx.h"

struct Trivial {
  int X;
};

struct NonTrivial {
  int *P;
  NonTrivial(const NonTrivial &Other);
  NonTrivial(NonTrivial &&Other);
};

void copy_from_itself_trivial(Trivial *Dst) {
  ::new (Dst) Trivial(*Dst); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void copy_from_itself(NonTrivial *Dst) {
  ::new (Dst) NonTrivial(*Dst); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void move_from_itself(NonTrivial *Dst) {
  ::new (Dst) NonTrivial(static_cast<NonTrivial &&>(*Dst)); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void copy_through_alias(NonTrivial *Dst) {
  NonTrivial *Alias = Dst; // expected-note{{'Alias' initialized here}}
  ::new (Dst) NonTrivial(*Alias); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void copy_from_local(const NonTrivial &Src) {
  NonTrivial Obj(Src); // expected-note{{'Obj' initialized here}}
  ::new (&Obj) NonTrivial(Obj); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

struct Outer {
  Trivial Inner;
};

void copy_from_enclosing_object(Outer *O) {
  ::new (&O->Inner) Trivial(O->Inner); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void copy_from_other(NonTrivial *Dst, NonTrivial *Src) {
  ::new (Dst) NonTrivial(*Src); // no-warning
}

void copy_to_previous_element(Trivial *P) {
  ::new (P - 1) Trivial(*P); // no-warning
}

void copy_to_next_element(NonTrivial *P) {
  ::new (P + 1) NonTrivial(*P); // no-warning
}

void copy_to_partially_overlapping_bytes(Trivial *P) {
  ::new ((char *)P + 2) Trivial(*P); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

struct Pair {
  Trivial A;
  Trivial B;
};

void copy_from_sibling_field(Pair *P) {
  ::new (&P->A) Trivial(P->B); // no-warning
}

struct Node {
  Node *Next;
  explicit Node(Node *N) : Next(N) {}
};

void self_referential_pointer(Node *Storage) {
  ::new (Storage) Node(Storage); // no-warning
}

void regular_new(NonTrivial *Src) {
  NonTrivial *P = new NonTrivial(*Src); // no-warning
  (void)P;
}

struct Memberwise {
  int X, Y;
  Memberwise(const Memberwise &O) : X(O.X), Y(O.Y) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void memberwise_copy(Memberwise *Dst) {
  ::new (Dst) Memberwise(*Dst); // expected-note{{Calling copy constructor for 'Memberwise'}}
}

struct IgnoresSource {
  int Field;
  IgnoresSource(IgnoresSource &) { Field = 0; }
};

void ignores_source(IgnoresSource *Dst) {
  ::new (Dst) IgnoresSource(*Dst); // no-warning
}

struct ReadsOwnField {
  int X, Y;
  ReadsOwnField(const ReadsOwnField &) : X(1), Y(X + 1) {}
};

void reads_own_field(ReadsOwnField *Dst) {
  ::new (Dst) ReadsOwnField(*Dst); // no-warning
}

struct ChainedReset {
  int *Begin, *End;
  ChainedReset(const ChainedReset &) { Begin = End = nullptr; }
};

void chained_assignment_in_self_copy(ChainedReset *Dst) {
  ::new (Dst) ChainedReset(*Dst); // no-warning
}

struct Ends {
  int Values[2];
  int &get(int I) { return Values[I]; }
  Ends(const Ends &) : Values{1, 2} { get(0) += 1; }
};

void member_call_on_this_in_self_copy(Ends *Dst) {
  ::new (Dst) Ends(*Dst); // no-warning
}

struct FromInt {
  int Field;
  explicit FromInt(int &V) : Field(V + 1) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void reads_member_by_reference(FromInt *Dst) {
  ::new (Dst) FromInt(Dst->Field); // expected-note{{Calling constructor for 'FromInt'}}
}

void construct_at_trivial(Trivial *Dst) {
  std::construct_at(Dst, *Dst); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

struct SwappedInStdlib {
  int X, Y;
  SwappedInStdlib(const SwappedInStdlib &O) : X(O.Y), Y(O.X) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void construct_at_swapped(SwappedInStdlib *Dst) {
  std::construct_at(Dst, *Dst); // expected-note{{Calling 'construct_at<SwappedInStdlib, SwappedInStdlib &>'}}
  // expected-note@Inputs/system-header-simulator-cxx.h:* {{Calling copy constructor for 'SwappedInStdlib'}}
}

void construct_at_other(Trivial *Dst, Trivial *Src) {
  std::construct_at(Dst, *Src); // no-warning
}

void array_trivial(Trivial *Dst) {
  ::new (Dst) Trivial[1]{*Dst}; // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void array_second_element(Trivial *Dst) {
  ::new (Dst) Trivial[2]{Dst[1], Dst[0]}; // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void array_other(Trivial *Dst, Trivial *Src) {
  ::new (Dst) Trivial[1]{*Src}; // no-warning
}

struct Agg {
  int X;
  int Y;
};

void aggregate_same_order(Agg *Dst) {
  ::new (Dst) Agg{Dst->X, Dst->Y}; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
}

struct Nested {
  Agg Inner;
  int Z;
};

void nested_aggregate(Nested *Dst) {
  ::new (Dst) Nested{{Dst->Z, Dst->Inner.X}, 0}; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
}

void aggregate_copy(Agg *Dst) {
  ::new (Dst) Agg{*Dst}; // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

struct Recursive;
void place(Recursive *Dst, const Recursive &Src, Recursive *Scratch,
           const Recursive *Other, int Depth);

struct Recursive {
  int X, Y;
  Recursive(const Recursive &O, Recursive *Scratch, const Recursive *Other,
            int Depth) {
    if (Depth) // expected-note{{'Depth' is 1}} expected-note{{Taking true branch}}
      place(Scratch, *Other, nullptr, nullptr, 0);
    X = O.Y; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
    Y = O.X;
  }
};

void place(Recursive *Dst, const Recursive &Src, Recursive *Scratch,
           const Recursive *Other, int Depth) {
  ::new (Dst) Recursive(Src, Scratch, Other, Depth); // expected-note{{Calling constructor for 'Recursive'}}
}

void same_new_expression_in_nested_frame(Recursive *Dst, Recursive *Scratch,
                                         const Recursive *Other) {
  place(Dst, *Dst, Scratch, Other, 1); // expected-note{{Calling 'place'}}
}

template <typename Derived> struct ReinitMixin {
  void reinit(const Derived &Source) {
    ::new (static_cast<Derived *>(this)) Derived(Source); // expected-note{{Calling implicit copy constructor for 'WithString'}}
  }
};

struct WithString : ReinitMixin<WithString> { // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  int Value;
  std::string Name;
  WithString(int V, std::string N) : Value(V), Name(N) {}
};

void crtp_reinit_from_itself(WithString &Obj) {
  Obj.reinit(Obj); // expected-note{{Calling 'ReinitMixin::reinit'}}
}

template <typename T> struct Box {
  T Value;
  Box(const Box &O) : Value(O.Value) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void class_template_non_trivial_member(Box<NonTrivial> *Dst) {
  ::new (Dst) Box<NonTrivial>(*Dst); // expected-note{{Calling copy constructor for 'Box<NonTrivial>'}}
}

struct Base1 {
  int A;
};
struct Base2 {
  int B;
};

struct TwoBases : Base1, Base2 {
  int X;
  TwoBases(const TwoBases &O) : Base1(O), Base2(O), X(O.X) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void bases_copied_from_itself(TwoBases *Dst) {
  ::new (Dst) TwoBases(*Dst); // expected-note{{Calling copy constructor for 'TwoBases'}}
}

struct Empty {};

struct EmptyBase : Empty {
  int X;
  EmptyBase(const EmptyBase &O)
      : Empty(O),
        X(O.X) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void empty_base_is_not_read(EmptyBase *Dst) {
  ::new (Dst) EmptyBase(*Dst); // expected-note{{Calling copy constructor for 'EmptyBase'}}
}

struct WritesOther {
  int X;
  explicit WritesOther(int &Other) {
    Other = 1; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
    X = Other;
  }
};

void larger_buffer(void *Buf) {
  ::new (Buf) WritesOther(*(int *)((char *)Buf + 64)); // no-warning
}

void aliases_the_object(void *Buf) {
  ::new (Buf) WritesOther(*(int *)Buf); // expected-note{{Calling constructor for 'WritesOther'}}
}

struct ThisAliasHelper;
int getX(const ThisAliasHelper &S);

struct ThisAliasHelper {
  int X, Y;
  ThisAliasHelper(const ThisAliasHelper &) {
    X = 1;
    Y = getX(*this) + 1;
  }
};

int getX(const ThisAliasHelper &S) { return S.X; }

void this_passed_to_helper(ThisAliasHelper *Dst) {
  ::new (Dst) ThisAliasHelper(*Dst); // no-warning
}

struct LocalThisAlias {
  int X, Y;
  LocalThisAlias(const LocalThisAlias &) {
    X = 1;
    auto *Self = this;
    Y = Self->X + 1;
  }
};

void this_stored_in_local(LocalThisAlias *Dst) {
  ::new (Dst) LocalThisAlias(*Dst); // no-warning
}

struct SourceByPointer {
  int X, Y;
  explicit SourceByPointer(const SourceByPointer *O) : X(O->Y), Y(O->X) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void source_by_pointer(SourceByPointer *Dst) {
  ::new (Dst) SourceByPointer(Dst); // expected-note{{Calling constructor for 'SourceByPointer'}}
}

struct SourceToHelper;
int getY(const SourceToHelper &S);

struct SourceToHelper {
  int X, Y;
  SourceToHelper(const SourceToHelper &O) {
    X = 1;
    Y = getY(O); // expected-note{{Calling 'getY'}}
  }
};

int getY(const SourceToHelper &S) { return S.Y; } // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}

void source_passed_to_helper(SourceToHelper *Dst) {
  ::new (Dst) SourceToHelper(*Dst); // expected-note{{Calling copy constructor for 'SourceToHelper'}}
}

struct SourceInLocal {
  int X, Y;
  SourceInLocal(const SourceInLocal &O) {
    X = 1;
    const SourceInLocal *Src = &O;
    Y = Src->Y; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void source_stored_in_local(SourceInLocal *Dst) {
  ::new (Dst) SourceInLocal(*Dst); // expected-note{{Calling copy constructor for 'SourceInLocal'}}
}

struct LocalReassigned {
  int X, Y;
  LocalReassigned(const LocalReassigned &O, const LocalReassigned &Other) {
    const LocalReassigned *P = &O;
    P = &Other;
    X = P->Y;
    Y = 0;
  }
};

void local_reassigned_to_other(LocalReassigned *Dst, LocalReassigned *Other) {
  ::new (Dst) LocalReassigned(*Dst, *Other); // no-warning
}

struct MethodOnSource {
  int X, Y;
  int getX() const { return X; } // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  MethodOnSource(const MethodOnSource &O) {
    X = 1;
    Y = O.getX(); // expected-note{{Calling 'MethodOnSource::getX'}}
  }
};

void method_called_on_source(MethodOnSource *Dst) {
  ::new (Dst) MethodOnSource(*Dst); // expected-note{{Calling copy constructor for 'MethodOnSource'}}
}

struct LambdaCapture {
  int X, Y;
  LambdaCapture(const LambdaCapture &O) {
    X = 1;
    Y = [&] { return O.Y; }(); // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void source_captured_by_lambda(LambdaCapture *Dst) {
  ::new (Dst) LambdaCapture(*Dst); // expected-note{{Calling copy constructor for 'LambdaCapture'}}
}

struct LambdaCapturesPointer {
  int X;
  const LambdaCapturesPointer *Next;
  explicit LambdaCapturesPointer(const LambdaCapturesPointer *O) : X(0) {
    Next = [O] { return O; }();
  }
};

void pointer_captured_by_lambda(LambdaCapturesPointer *Dst) {
  ::new (Dst) LambdaCapturesPointer(Dst); // no-warning
}

struct Conditional {
  int X, Y;
  Conditional(const Conditional &O, const Conditional &Other, bool B) {
    X = 1;
    const Conditional &R = B ? O : Other; // expected-note{{'B' is true}} expected-note{{'?' condition is true}}
    Y = R.Y; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void source_chosen_by_conditional(Conditional *Dst, Conditional *Other) {
  ::new (Dst) Conditional(*Dst, *Other, true); // expected-note{{Calling constructor for 'Conditional'}}
}

void other_chosen_by_conditional(Conditional *Dst, Conditional *Other) {
  ::new (Dst) Conditional(*Dst, *Other, false); // no-warning
}

struct AddressOfSource {
  int X, Y;
  AddressOfSource(const AddressOfSource &O) {
    X = 1;
    const AddressOfSource *P = __builtin_addressof(O);
    Y = P->Y; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void address_of_source(AddressOfSource *Dst) {
  ::new (Dst) AddressOfSource(*Dst); // expected-note{{Calling copy constructor for 'AddressOfSource'}}
}

struct OpaqueMethodOnSource {
  int X, Y;
  int get() const;
  OpaqueMethodOnSource(const OpaqueMethodOnSource &O) {
    X = 1;
    Y = O.get(); // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void opaque_method_called_on_source(OpaqueMethodOnSource *Dst) {
  ::new (Dst) OpaqueMethodOnSource(*Dst); // expected-note{{Calling copy constructor for 'OpaqueMethodOnSource'}}
}

struct StoresAddress;
const StoresAddress *Remembered;
void remember(const StoresAddress &R) { Remembered = &R; }

struct StoresAddress {
  int X;
  StoresAddress(const StoresAddress &O) : X(0) { remember(O); }
};

void callee_stores_address_of_source(StoresAddress *Dst) {
  ::new (Dst) StoresAddress(*Dst); // no-warning
}

struct ReadsStoredAddress;
const ReadsStoredAddress *RememberedToRead;
void rememberToRead(const ReadsStoredAddress &R) { RememberedToRead = &R; }

struct ReadsStoredAddress {
  int X, Y;
  ReadsStoredAddress(const ReadsStoredAddress &O) {
    X = 1;
    rememberToRead(O);
    Y = RememberedToRead->X; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void read_through_stored_address(ReadsStoredAddress *Dst) {
  ::new (Dst) ReadsStoredAddress(*Dst); // expected-note{{Calling copy constructor for 'ReadsStoredAddress'}}
}

struct MovedFrom {
  int X, Y;
  MovedFrom(MovedFrom &&O) : X(O.Y), Y(O.X) {} // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void move_from_itself_with_std_move(MovedFrom *Dst) {
  ::new (Dst) MovedFrom(std::move(*Dst)); // expected-note{{Calling move constructor for 'MovedFrom'}}
}

struct Pool {
  int Count;
  alignas(8) char Buf[64];
  void note();
  int peek() const { return Buf[0]; } // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
};

struct PoolItem {
  int V;
  explicit PoolItem(Pool &P) : V(P.Count) { P.note(); }
};

PoolItem *pool_method_not_inlined(Pool &P) {
  return ::new (P.Buf) PoolItem(P); // no-warning
}

struct PoolItemLambda {
  int V;
  explicit PoolItemLambda(Pool &P) : V(0) {
    [&] { P.note(); }();
  }
};

PoolItemLambda *pool_captured_by_lambda(Pool &P) {
  return ::new (P.Buf) PoolItemLambda(P); // no-warning
}

struct PoolItemPeek {
  int V;
  explicit PoolItemPeek(Pool &P) : V(P.peek()) {} // expected-note{{Calling 'Pool::peek'}}
};

PoolItemPeek *pool_reads_buffer(Pool &P) {
  return ::new (P.Buf) PoolItemPeek(P); // expected-note{{Calling constructor for 'PoolItemPeek'}}
}

struct ArithmeticOnMember {
  int A[4];
  int S;
  ArithmeticOnMember(const ArithmeticOnMember &O) : S(0) {
    const int *P = O.A + 1;
    S = *P; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void read_after_pointer_arithmetic(ArithmeticOnMember *Dst) {
  ::new (Dst) ArithmeticOnMember(*Dst); // expected-note{{Calling copy constructor for 'ArithmeticOnMember'}}
}

struct IncrementsSource {
  int A[4];
  int S;
  IncrementsSource(const IncrementsSource &O) : S(0) {
    const int *P = O.A;
    ++P;
    S = *P; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void read_after_increment(IncrementsSource *Dst) {
  ::new (Dst) IncrementsSource(*Dst); // expected-note{{Calling copy constructor for 'IncrementsSource'}}
}

struct LambdaDereferencesPointer {
  int X, Y;
  explicit LambdaDereferencesPointer(const LambdaDereferencesPointer *O)
      : X(1) {
    Y = [O] { return O->Y; }(); // expected-note{{Calling 'operator()'}}
  } // expected-warning@-1{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note@-1{{Placement new accesses the storage it is placed into during construction of the new object}}
};

void pointer_dereferenced_in_lambda(LambdaDereferencesPointer *Dst) {
  ::new (Dst) LambdaDereferencesPointer(Dst); // expected-note{{Calling constructor for 'LambdaDereferencesPointer'}}
}

extern "C" void *memcpy(void *, const void *, __SIZE_TYPE__);

struct CopiesBytes {
  int X, Y;
  CopiesBytes(const CopiesBytes &O) {
    memcpy((void *)this, &O, sizeof(CopiesBytes)); // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void memcpy_from_source(CopiesBytes *Dst) {
  ::new (Dst) CopiesBytes(*Dst); // expected-note{{Calling copy constructor for 'CopiesBytes'}}
}

struct MemberAddress {
  int X;
  int *P;
  MemberAddress(const MemberAddress &O) : X(1), P(const_cast<int *>(&O.X)) {}
};

void address_of_member_is_not_read(MemberAddress *Dst) {
  ::new (Dst) MemberAddress(*Dst); // no-warning
}

struct ShiftsPointer {
  ShiftsPointer *Next;
  explicit ShiftsPointer(ShiftsPointer *N) : Next(N + 1) {}
};

void pointer_arithmetic_is_not_read(ShiftsPointer *Dst) {
  ::new (Dst) ShiftsPointer(Dst); // no-warning
}

struct MethodWithoutRead {
  int X;
  int answer() const { return 42; }
  MethodWithoutRead(const MethodWithoutRead &O) : X(O.answer()) {}
};

void method_that_reads_nothing(MethodWithoutRead *Dst) {
  ::new (Dst) MethodWithoutRead(*Dst); // no-warning
}

struct DataOnSource {
  int A[4];
  int S;
  const int *data() const { return A; }
  DataOnSource(const DataOnSource &O) : S(0) {
    const int *P = O.data();
    S = P[1]; // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}}
  }
};

void read_through_returned_pointer(DataOnSource *Dst) {
  ::new (Dst) DataOnSource(*Dst); // expected-note{{Calling copy constructor for 'DataOnSource'}}
}

struct ThisInLambda {
  int X, Y;
  int get() const {
    return [this] { return X; }(); // expected-warning{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Placement new accesses the storage it is placed into during construction of the new object}} expected-note{{Calling 'operator()'}}
  }
  ThisInLambda(const ThisInLambda &O) : X(1) {
    Y = O.get(); // expected-note{{Calling 'ThisInLambda::get'}}
  }
};

void this_captured_in_method_on_source(ThisInLambda *Dst) {
  ::new (Dst) ThisInLambda(*Dst); // expected-note{{Calling copy constructor for 'ThisInLambda'}}
}

#endif
