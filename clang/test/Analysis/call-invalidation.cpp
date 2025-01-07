// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

template <class T> void clang_analyzer_dump(T);
void clang_analyzer_eval(bool);

void usePointer(int * const *);
void useReference(int * const &);

void testPointer() {
  int x;
  int *p;

  p = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  usePointer(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  p = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useReference(p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  int * const cp1 = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  usePointer(&cp1);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  int * const cp2 = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useReference(cp2);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


struct Wrapper {
  int *ptr;
};

void useStruct(Wrapper &w);
void useConstStruct(const Wrapper &w);

void testPointerStruct() {
  int x;
  Wrapper w;

  w.ptr = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  w.ptr = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useConstStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


struct RefWrapper {
  int &ref;
};

void useStruct(RefWrapper &w);
void useConstStruct(const RefWrapper &w);

void testReferenceStruct() {
  int x;
  RefWrapper w = { x };

  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}

// FIXME: This test is split into two functions because region invalidation
// does not preserve reference bindings.
void testConstReferenceStruct() {
  int x;
  RefWrapper w = { x };

  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useConstStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


int usePointerPure(int * const *) __attribute__((pure));
int usePointerConst(int * const *) __attribute__((const));

void testPureConst() {
  extern int global;
  int x;
  int *p;

  p = &x;
  x = 42;
  global = -5;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  (void)usePointerPure(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  (void)usePointerConst(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointer(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(global == -5); // expected-warning{{UNKNOWN}}
}


struct PlainStruct {
  int x, y;
  mutable int z;
};

PlainStruct glob;

void useAnything(void *);
void useAnythingConst(const void *);

void testInvalidationThroughBaseRegionPointer() {
  PlainStruct s1;
  s1.x = 1;
  s1.z = 1;
  clang_analyzer_eval(s1.x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(s1.z == 1); // expected-warning{{TRUE}}
  // Not only passing a structure pointer through const pointer parameter,
  // but also passing a field pointer through const pointer parameter
  // should preserve the contents of the structure.
  useAnythingConst(&(s1.y));
  clang_analyzer_eval(s1.x == 1); // expected-warning{{TRUE}}
  // FIXME: Should say "UNKNOWN", because it is not uncommon to
  // modify a mutable member variable through const pointer.
  clang_analyzer_eval(s1.z == 1); // expected-warning{{TRUE}}
  useAnything(&(s1.y));
  clang_analyzer_eval(s1.x == 1); // expected-warning{{UNKNOWN}}
}


void useFirstConstSecondNonConst(const void *x, void *y);
void useFirstNonConstSecondConst(void *x, const void *y);

void testMixedConstNonConstCalls() {
  PlainStruct s2;
  s2.x = 1;
  useFirstConstSecondNonConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.x == 1); // expected-warning{{UNKNOWN}}
  s2.x = 1;
  useFirstNonConstSecondConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.x == 1); // expected-warning{{UNKNOWN}}
  s2.y = 1;
  useFirstConstSecondNonConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.y == 1); // expected-warning{{UNKNOWN}}
  s2.y = 1;
  useFirstNonConstSecondConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.y == 1); // expected-warning{{UNKNOWN}}
}

namespace std {
class Opaque {
public:
  Opaque();
  int nested_member;
};
} // namespace std

struct StdWrappingOpaque {
  std::Opaque o; // first member
  int uninit;
};
struct StdWrappingOpaqueSwapped {
  int uninit; // first member
  std::Opaque o;
};

int testStdCtorDoesNotInvalidateParentObject() {
  StdWrappingOpaque obj;
  int x = obj.o.nested_member; // no-garbage: std::Opaque::ctor might initialized this
  int y = obj.uninit; // FIXME: We should have a garbage read here. Read the details.
  // As the first member ("obj.o") is invalidated, a conjured default binding is bound
  // to the offset 0 within cluster "obj", and this masks every uninitialized fields
  // that follows. We need a better store with extents to fix this.
  return x + y;
}

int testStdCtorDoesNotInvalidateParentObjectSwapped() {
  StdWrappingOpaqueSwapped obj;
  int x = obj.o.nested_member; // no-garbage: std::Opaque::ctor might initialized this
  int y = obj.uninit; // expected-warning {{Assigned value is garbage or undefined}}
  return x + y;
}

class UserProvidedOpaque {
public:
  UserProvidedOpaque(); // might reinterpret_cast(this)
  int nested_member;
};

struct WrappingUserProvidedOpaque {
  UserProvidedOpaque o; // first member
  int uninit;
};
struct WrappingUserProvidedOpaqueSwapped {
  int uninit; // first member
  UserProvidedOpaque o;
};

int testUserProvidedCtorInvalidatesParentObject() {
  WrappingUserProvidedOpaque obj;
  int x = obj.o.nested_member; // no-garbage: UserProvidedOpaque::ctor might initialized this
  int y = obj.uninit; // no-garbage: UserProvidedOpaque::ctor might reinterpret_cast(this) and write to the "uninit" member.
  return x + y;
}

int testUserProvidedCtorInvalidatesParentObjectSwapped() {
  WrappingUserProvidedOpaqueSwapped obj;
  int x = obj.o.nested_member; // no-garbage: same as above
  int y = obj.uninit; // no-garbage: same as above
  return x + y;
}

struct WrappingStdWrappingOpaqueOuterInits {
  int first = 1;
  std::Opaque second;
  int third = 3;
  WrappingStdWrappingOpaqueOuterInits() {
    clang_analyzer_dump(first); // expected-warning {{1 S32b}}
    clang_analyzer_dump(second.nested_member); // expected-warning {{derived_}}
    clang_analyzer_dump(third); // expected-warning {{3 S32b}}
  }
};

struct WrappingUserProvidedOpaqueOuterInits {
  int first = 1; // Potentially overwritten by UserProvidedOpaque::ctor
  UserProvidedOpaque second; // Invalidates the object so far.
  int third = 3; // Happens after UserProvidedOpaque::ctor, thus preserved!
  WrappingUserProvidedOpaqueOuterInits() {
    clang_analyzer_dump(first); // expected-warning {{derived_}}
    clang_analyzer_dump(second.nested_member); // expected-warning {{derived_}}
    clang_analyzer_dump(third); // expected-warning {{3 S32b}}
  }
};

extern "C++" {
namespace std {
inline namespace v1 {
namespace custom_ranges {
struct Fancy {
struct iterator {
struct Opaque {
  Opaque();
  int nested_member;
}; // struct Opaque
}; // struct iterator
}; // struct Fancy
} // namespace custom_ranges
} // namespace v1
} // namespace std
} // extern "C++"

struct StdWrappingFancyOpaque {
  int uninit;
  std::custom_ranges::Fancy::iterator::Opaque o;
};

int testNestedStdNamespacesAndRecords() {
  StdWrappingFancyOpaque obj;
  int x = obj.o.nested_member; // no-garbage: ctor
  int y = obj.uninit; // expected-warning {{Assigned value is garbage or undefined}}
  return x + y;
}
