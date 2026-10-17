// RUN: %clang_analyze_cc1 -std=c++20 %s \
// RUN:   -analyzer-checker=core,alpha.cplusplus.PlacementSelfCopy \
// RUN:   -analyzer-output=text -verify -triple x86_64-unknown-linux-gnu

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

struct Deep {
  int *P;
  int N;
  Deep(const Deep &O) : P(new int[O.N]), N(O.N) { P[0] = O.P[0]; } // expected-note{{Part of the storage is overwritten here}} expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
};

void deep_copy_reads_overwritten_pointer(Deep *Dst) {
  ::new (Dst) Deep(*Dst); // expected-note{{Calling copy constructor for 'Deep'}}
}

struct Swapped {
  int X, Y;
  Swapped(const Swapped &O) : X(O.Y), Y(O.X) {} // expected-note{{Part of the storage is overwritten here}} expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
};

void swapped_copy(Swapped *Dst) {
  ::new (Dst) Swapped(*Dst); // expected-note{{Calling copy constructor for 'Swapped'}}
}

struct Memberwise {
  int X, Y;
  Memberwise(const Memberwise &O) : X(O.X), Y(O.Y) {}
};

void memberwise_copy(Memberwise *Dst) {
  ::new (Dst) Memberwise(*Dst); // no-warning
}

struct ReadsOwnField {
  int X, Y;
  ReadsOwnField(const ReadsOwnField &O) : X(O.X), Y(X + 1) {}
};

void reads_own_field(ReadsOwnField *Dst) {
  ::new (Dst) ReadsOwnField(*Dst); // no-warning
}

struct IgnoresSource {
  int Field;
  IgnoresSource(IgnoresSource &) { Field = 0; }
};

void ignores_source(IgnoresSource *Dst) {
  ::new (Dst) IgnoresSource(*Dst); // no-warning
}

struct FromInt {
  int Field;
  explicit FromInt(int &V) : Field(V + 1) {}
};

void reads_before_writing(FromInt *Dst) {
  ::new (Dst) FromInt(Dst->Field); // no-warning
}

struct HoldsTrivial {
  Trivial T;
  int K;
  HoldsTrivial(const HoldsTrivial &O) : T(O.T), K(O.K) {}
};

void member_copied_before_overwrite(HoldsTrivial *Dst) {
  ::new (Dst) HoldsTrivial(*Dst); // no-warning
}

void construct_at_trivial(Trivial *Dst) {
  std::construct_at(Dst, *Dst); // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

struct SwappedInStdlib {
  int X, Y;
  SwappedInStdlib(const SwappedInStdlib &O) : X(O.Y), Y(O.X) {} // expected-note{{Part of the storage is overwritten here}} expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
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

void array_deep(Deep *Dst) {
  ::new (Dst) Deep[1]{*Dst}; // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

void array_second_element(Trivial *Dst) {
  ::new (Dst) Trivial[2]{Dst[1], Dst[0]}; // expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
}

void array_other(Trivial *Dst, Trivial *Src) {
  ::new (Dst) Trivial[1]{*Src}; // no-warning
}

struct Agg {
  int X;
  int Y;
};

void aggregate_swapped(Agg *Dst) {
  ::new (Dst) Agg{Dst->Y, Dst->X}; // expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
}

void aggregate_same_order(Agg *Dst) {
  ::new (Dst) Agg{Dst->X, Dst->Y}; // no-warning
}

struct Nested {
  Agg Inner;
  int Z;
};

void nested_aggregate(Nested *Dst) {
  ::new (Dst) Nested{{Dst->Z, Dst->Inner.X}, 0}; // expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
}

void aggregate_copy(Agg *Dst) {
  ::new (Dst) Agg{*Dst}; // expected-warning{{Object constructed by placement new is initialized from the storage it is placed into}} expected-note{{Object constructed by placement new is initialized from the storage it is placed into}}
}

struct ChainedReset {
  int *Begin, *End;
  ChainedReset(const ChainedReset &) { Begin = End = nullptr; }
};

void chained_assignment_in_self_copy(ChainedReset *Dst) {
  ::new (Dst) ChainedReset(*Dst); // no-warning
}

struct Stream {
  int *Cur, *End;
  explicit Stream(int &Source) { Cur = End = nullptr; (void)Source; }
};

void chained_assignment_from_other(Stream *Dst, int &Other) {
  ::new (Dst) Stream(Other); // no-warning
}

struct Ends {
  int Values[2];
  int &get(int I) { return Values[I]; }
  explicit Ends(const int (&Init)[2]) : Values{Init[0], Init[1]} {
    get(0) += 1;
  }
};

void member_call_from_other(Ends *Dst, const int (&Init)[2]) {
  ::new (Dst) Ends(Init); // no-warning
}

struct Recursive;
void place(Recursive *Dst, const Recursive &Src, Recursive *Scratch,
           const Recursive *Other, int Depth);

struct Recursive {
  int X, Y;
  Recursive(const Recursive &O, Recursive *Scratch, const Recursive *Other,
            int Depth)
      : X(O.Y) { // expected-note{{Part of the storage is overwritten here}}
    if (Depth) // expected-note{{'Depth' is 1}} expected-note{{Taking true branch}}
      place(Scratch, *Other, nullptr, nullptr, 0);
    Y = O.X; // expected-warning{{Placement new reads a part of its storage that it has already overwritten}} expected-note{{Placement new reads a part of its storage that it has already overwritten}}
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
