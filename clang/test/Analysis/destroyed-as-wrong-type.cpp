// RUN: %clang_analyze_cc1 -std=c++11 %s \
// RUN:   -analyzer-checker=core,cplusplus.DestroyedAsWrongType \
// RUN:   -analyzer-output=text -verify -triple x86_64-unknown-linux-gnu
// RUN: %clang_analyze_cc1 -std=c++20 %s \
// RUN:   -analyzer-checker=core,cplusplus.DestroyedAsWrongType \
// RUN:   -analyzer-output=text -verify -triple x86_64-unknown-linux-gnu

#include "Inputs/system-header-simulator-cxx.h"

struct A {};

struct B {
  ~B();
};

void consume(void *);

void replace_local() {
  B b;
  new (&b) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

void replace_heap() {
  B *b = new B;
  new (b) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  delete b;
  // expected-warning@-1 {{Deleting an object of type 'A' through a pointer to 'B'}}
  // expected-note@-2    {{Deleting an object of type 'A' through a pointer to 'B'}}
}

struct Base {
  ~Base();
};
struct Derived : Base {};

void delete_through_base_without_virtual_dtor() {
  Base *p = new Base;
  new (p) Derived;
  // expected-note@-1    {{Object of type 'Derived' is created here by placement new}}
  delete p;
  // expected-warning@-1 {{Deleting an object of type 'Derived' through a pointer to 'Base'}}
  // expected-note@-2    {{Deleting an object of type 'Derived' through a pointer to 'Base'}}
}

void new_derived_deleted_through_base_without_virtual_dtor() {
  Base *p = new Derived;
  delete p; // no-warning
}

struct Holder {
  // expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  B Member;
};

void replace_member() {
  Holder h;
  new (&h.Member) A;
  // expected-note@-1 {{Object of type 'A' is created here by placement new}}
}
// expected-note@-1   {{Calling implicit destructor for 'Holder'}}

void destroy_then_replace() {
  B b;
  b.~B();
  new (&b) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

void explicit_destructor_on_replaced_storage() {
  B b;
  new (&b) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  b.~B();
  // expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
}

void restore_after_destroying_placed_object() {
  B b;
  b.~B();
  new (&b) A;
  reinterpret_cast<A *>(&b)->~A();
  new (&b) B;
} // no-warning

void reconstruct_original_type() {
  B b;
  b.~B();
  new (&b) B;
} // no-warning

void reconstruct_on_heap() {
  B *b = new B;
  b->~B();
  new (b) B;
  delete b; // no-warning
}

struct VBase {
  virtual ~VBase();
};
struct VDerived : VBase {};

void delete_through_base_with_virtual_dtor() {
  VBase *p = new VBase;
  p->~VBase();
  new (p) VDerived;
  delete p; // no-warning
}

void array_delete_through_base_with_virtual_dtor() {
  VBase *p = new VBase[1];
  p->~VBase();
  new (p) VDerived;
  // expected-note@-1    {{Object of type 'VDerived' is created here by placement new}}
  delete[] p;
  // expected-warning@-1 {{Deleting an object of type 'VDerived' through a pointer to 'VBase'}}
  // expected-note@-2    {{Deleting an object of type 'VDerived' through a pointer to 'VBase'}}
}

void explicit_virtual_destructor_through_base() {
  VBase *p = new VBase;
  p->~VBase();
  new (p) VDerived;
  p->~VBase(); // no-warning
}

struct BaseWithData {
  int X;
  ~BaseWithData();
};
struct DerivedFromBaseWithData : BaseWithData {};

void explicit_base_destructor_on_derived_object() {
  BaseWithData b;
  b.~BaseWithData();
  auto *D = new (&b) DerivedFromBaseWithData;
  static_cast<BaseWithData *>(D)->~BaseWithData();
  new (&b) BaseWithData;
} // no-warning

void explicit_base_destructor_without_restoring() {
  BaseWithData b;
  b.~BaseWithData();
  auto *D = new (&b) DerivedFromBaseWithData;
  // expected-note@-1 {{Object of type 'DerivedFromBaseWithData' is created here by placement new}}
  static_cast<BaseWithData *>(D)->~BaseWithData();
}
// expected-warning@-1 {{Destructor of 'BaseWithData' is called on storage that holds an object of type 'DerivedFromBaseWithData'}}
// expected-note@-2    {{Destructor of 'BaseWithData' is called on storage that holds an object of type 'DerivedFromBaseWithData'}}

void storage_passed_to_opaque_function() {
  B b;
  new (&b) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
  consume(&b);
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

struct WithDtor {
  ~WithDtor();
};
struct DerivedFromWithDtor : WithDtor {};

void raw_memory_destroyed_through_base() {
  void *Raw = ::operator new(sizeof(DerivedFromWithDtor));
  WithDtor *P = new (Raw) DerivedFromWithDtor;
  P->~WithDtor();
  ::operator delete(Raw);
} // no-warning

void raw_local_buffer() {
  alignas(B) unsigned char Buf[sizeof(B)];
  B *P = new (Buf) B;
  P->~B();
} // no-warning

void raw_local_buffer_destroyed_as_other_type() {
  alignas(B) unsigned char Buf[sizeof(B)];
  new (Buf) A;
  reinterpret_cast<B *>(Buf)->~B();
} // no-warning

struct CtorBase {
  CtorBase() {}
};
struct WithCtor : CtorBase {
  WithCtor(int) {}
  ~WithCtor();
};
struct InheritsCtor : WithCtor {
  using WithCtor::WithCtor;
};

void inherited_constructor() {
  InheritsCtor I(0);
} // no-warning

union OneOf {
  B First;
  A Second;
  OneOf() { new (this) B; }
  ~OneOf();
};

void union_member_created_through_union_address() {
  OneOf U;
  U.~OneOf();
  new (&U) OneOf;
} // no-warning

struct Reassignable {
  Reassignable();
  Reassignable(const Reassignable &);
  ~Reassignable();
  Reassignable &operator=(const Reassignable &Other) {
    this->~Reassignable();
    new (this) Reassignable(Other);
    // expected-note@-1 {{Object of type 'Reassignable' is created here by placement new}}
    return *this;
  }
};
struct WrapsReassignable : Reassignable {};

void recreate_base_subobject(const WrapsReassignable &Other) {
  WrapsReassignable W;
  W = Other;
  // expected-note@-1  {{Calling implicit copy assignment operator for 'WrapsReassignable'}}
  // expected-note@-2  {{Calling copy assignment operator for 'Reassignable'}}
  // expected-note@-3  {{Returning from copy assignment operator for 'Reassignable'}}
  // expected-note@-4  {{Returning from copy assignment operator for 'WrapsReassignable'}}
}
// expected-warning@-1 {{Destructor of 'WrapsReassignable' is called on storage that holds an object of type 'Reassignable'}}
// expected-note@-2    {{Destructor of 'WrapsReassignable' is called on storage that holds an object of type 'Reassignable'}}

struct Trivial {
  int X;
};

void trivially_destructible_storage() {
  Trivial t;
  new (&t) A;
} // no-warning

struct WithPlacementNew {
  ~WithPlacementNew();
  static void *operator new(size_t, void *P) { return P; }
};

void restore_through_class_placement_new() {
  WithPlacementNew W;
  W.~WithPlacementNew();
  ::new (&W) A;
  new (&W) WithPlacementNew;
} // no-warning

struct WithOpaquePlacementNew {
  ~WithOpaquePlacementNew();
  static void *operator new(size_t, void *P);
};

void restore_through_opaque_class_placement_new() {
  WithOpaquePlacementNew W;
  W.~WithOpaquePlacementNew();
  ::new (&W) A;
  new (&W) WithOpaquePlacementNew;
} // no-warning

struct OffsetHolder {
  int X;
  B Member;
};

void restore_member_through_byte_offset() {
  OffsetHolder H;
  H.Member.~B();
  ::new (&H.Member) A;
  ::new (reinterpret_cast<char *>(&H) + __builtin_offsetof(OffsetHolder, Member)) B;
} // no-warning

struct PlacementTag {};
struct WithTaggedPlacementNew {
  ~WithTaggedPlacementNew();
  static void *operator new(size_t, PlacementTag, void *P) { return P; }
};

void restore_through_tagged_placement_new() {
  WithTaggedPlacementNew W;
  W.~WithTaggedPlacementNew();
  ::new (&W) A;
  new (PlacementTag{}, &W) WithTaggedPlacementNew;
} // no-warning

struct PlacedWithClassNew {
  static void *operator new(size_t, void *P) { return P; }
};

void replace_through_class_placement_new() {
  B b;
  new (&b) PlacedWithClassNew;
  // expected-note@-1  {{Object of type 'PlacedWithClassNew' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'PlacedWithClassNew'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'PlacedWithClassNew'}}

struct EscapingHolder {
  // expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  B Member;
};

void member_passed_to_opaque_function_with_parent() {
  EscapingHolder H;
  new (&H.Member) A;
  // expected-note@-1 {{Object of type 'A' is created here by placement new}}
  consume(&H);
}
// expected-note@-1   {{Calling implicit destructor for 'EscapingHolder'}}

struct MutatedHolder {
  // expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  B Member;
  void mutate();
};

void member_of_object_with_opaque_method() {
  MutatedHolder H;
  new (&H.Member) A;
  // expected-note@-1 {{Object of type 'A' is created here by placement new}}
  H.mutate();
}
// expected-note@-1   {{Calling implicit destructor for 'MutatedHolder'}}

struct HeapHolder {
  // expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  B Member;
};

void heap_member_passed_to_opaque_function_with_parent() {
  HeapHolder *H = new HeapHolder;
  new (&H->Member) A;
  // expected-note@-1 {{Object of type 'A' is created here by placement new}}
  consume(H);
  delete H;
  // expected-note@-1 {{Calling implicit destructor for 'HeapHolder'}}
}

struct OpaqueDtorHolder {
  B Member;
  ~OpaqueDtorHolder();
};

void member_destroyed_by_opaque_destructor() {
  OpaqueDtorHolder H;
  new (&H.Member) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

void member_restored_before_opaque_destructor() {
  OpaqueDtorHolder H;
  H.Member.~B();
  new (&H.Member) A;
  reinterpret_cast<A *>(&H.Member)->~A();
  new (&H.Member) B;
} // no-warning

void array_element() {
  B Arr[2];
  new (&Arr[1]) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

void first_array_element() {
  B Arr[2];
  new (&Arr[0]) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

void array_element_restored() {
  B Arr[2];
  Arr[1].~B();
  new (&Arr[1]) A;
  reinterpret_cast<A *>(&Arr[1])->~A();
  new (&Arr[1]) B;
} // no-warning

struct InlineDtor {
  int X;
  ~InlineDtor() {}
};

void multidimensional_array_element() {
  B M[2][3];
  new (&M[1][1]) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

void multidimensional_array_element_with_inline_destructor() {
  InlineDtor M[2][3];
  new (&M[1][1]) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'InlineDtor' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'InlineDtor' is called on storage that holds an object of type 'A'}}

void multidimensional_array_element_restored() {
  B M[2][3];
  M[1][1].~B();
  new (&M[1][1]) A;
  reinterpret_cast<A *>(&M[1][1])->~A();
  new (&M[1][1]) B;
} // no-warning

void heap_array_element() {
  B *Arr = new B[2];
  new (&Arr[1]) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  delete[] Arr;
  // expected-warning@-1 {{Deleting an object of type 'A' through a pointer to 'B'}}
  // expected-note@-2    {{Deleting an object of type 'A' through a pointer to 'B'}}
}

void heap_first_array_element() {
  B *Arr = new B[2];
  new (&Arr[0]) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  delete[] Arr;
  // expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
}

struct CastBase {
  ~CastBase();
};
struct CastDerived : CastBase {};

void recreate_base_through_reinterpret_cast() {
  CastDerived D;
  reinterpret_cast<CastBase *>(&D)->~CastBase();
  new (reinterpret_cast<CastBase *>(&D)) CastBase;
  // expected-note@-1  {{Object of type 'CastBase' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'CastDerived' is called on storage that holds an object of type 'CastBase'}}
// expected-note@-2    {{Destructor of 'CastDerived' is called on storage that holds an object of type 'CastBase'}}

void restore_through_array_placement_new() {
  B b;
  b.~B();
  new (&b) A;
  new (&b) B[1];
} // no-warning

void heap_destroy_then_replace() {
  B *P = new B;
  P->~B();
  new (P) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  delete P;
  // expected-warning@-1 {{Deleting an object of type 'A' through a pointer to 'B'}}
  // expected-note@-2    {{Deleting an object of type 'A' through a pointer to 'B'}}
}

void heap_recreate_base_through_reinterpret_cast() {
  CastDerived *P = new CastDerived;
  P->~CastDerived();
  new (reinterpret_cast<CastBase *>(P)) CastBase;
  // expected-note@-1    {{Object of type 'CastBase' is created here by placement new}}
  delete P;
  // expected-warning@-1 {{Deleting an object of type 'CastBase' through a pointer to 'CastDerived'}}
  // expected-note@-2    {{Deleting an object of type 'CastBase' through a pointer to 'CastDerived'}}
}

template <class T> void placeInto(void *P) {
  ::new (P) T;
  // expected-note@-1 {{Object of type 'A' is created here by placement new}}
}

void replace_from_template() {
  B b;
  placeInto<A>(&b);
  // expected-note@-1  {{Calling 'placeInto<A>'}}
  // expected-note@-2  {{Returning from 'placeInto<A>'}}
}
// expected-warning@-1 {{Destructor of 'B' is called on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called on storage that holds an object of type 'A'}}

namespace std {
template <class T> void destroy_at(T *P) { P->~T(); }
template <class T> struct default_delete {
  void operator()(T *P) const { delete P; }
};
template <class T> struct holder {
  T Value;
};
} // namespace std

void destroy_at_on_replaced_storage() {
  B b;
  new (&b) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  std::destroy_at(&b);
  // expected-warning@-1 {{Destructor of 'B' is called by 'std::destroy_at' on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called by 'std::destroy_at' on storage that holds an object of type 'A'}}
  new (&b) B;
}

void destroy_at_through_function_pointer() {
  B b;
  new (&b) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  void (*Destroy)(B *) = std::destroy_at<B>;
  Destroy(&b);
  // expected-warning@-1 {{Destructor of 'B' is called by 'std::destroy_at' on storage that holds an object of type 'A'}}
  // expected-note@-2    {{Destructor of 'B' is called by 'std::destroy_at' on storage that holds an object of type 'A'}}
  new (&b) B;
}

void delete_in_std_deleter() {
  B *P = new B;
  new (P) A;
  // expected-note@-1    {{Object of type 'A' is created here by placement new}}
  std::default_delete<B>()(P);
  // expected-warning@-1 {{Deleting an object of type 'A' through a pointer to 'B' by 'std::default_delete<B>::operator()'}}
  // expected-note@-2    {{Deleting an object of type 'A' through a pointer to 'B' by 'std::default_delete<B>::operator()'}}
}

void member_of_std_object() {
  std::holder<B> H;
  new (&H.Value) A;
  // expected-note@-1  {{Object of type 'A' is created here by placement new}}
}
// expected-warning@-1 {{Destructor of 'B' is called by 'std::holder<B>::~holder' on storage that holds an object of type 'A'}}
// expected-note@-2    {{Destructor of 'B' is called by 'std::holder<B>::~holder' on storage that holds an object of type 'A'}}

#if __cplusplus >= 202002L
namespace std {
struct destroying_delete_t {
  explicit destroying_delete_t() = default;
};
} // namespace std

struct WithDestroyingDelete {
  ~WithDestroyingDelete();
  void operator delete(WithDestroyingDelete *P, std::destroying_delete_t);
};

void destroying_operator_delete() {
  WithDestroyingDelete *P = new WithDestroyingDelete;
  new (P) A;
  delete P; // no-warning
}
#endif
