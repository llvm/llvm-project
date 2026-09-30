// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnborrowedLocalVarsChecker -verify %s

#include "mock-canborrow.h"

void someFunction();

Vector<char> makeVector();

Vector<char> *getVectorPtr();
Vector<char> &getVectorRef();

namespace loan_shapes {
void reference_loan(Vector<char> &vec) {
  char &c = vec[0];
  // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void pointer_loan(Vector<char> &vec) {
  char *p = vec.data();
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void view_loan(Vector<char> &vec) {
  StringView sv = vec.data();
  // expected-warning@-1{{Local variable 'sv' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void gsl_pointer_loan(Vector<char> &vec) {
  CharSpan s = makeSpan(vec);
  // expected-warning@-1{{Local variable 's' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void simple_container(SimpleContainer<char> &container) {
  char &c = container[0];
  // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'SimpleContainer<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void two_hops(Vector<char> &vec) {
  StringView sv = makeView(vec.data());
  // expected-warning@-1{{Local variable 'sv' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}
} // namespace loan_shapes

namespace not_a_loan {
void alias_reference(Vector<char> &vec) {
  Vector<char> &r = vec;
  someFunction();
  r.append('x');
}

void alias_const_reference(Vector<char> &vec) {
  const Vector<char> &r = vec;
  someFunction();
}

void alias_pointer(Vector<char> &vec) {
  Vector<char> *p = &vec;
  someFunction();
  p->append('x');
}

void value_copy(Vector<char> &vec) {
  char c = vec[0];
  someFunction();
  (void)c;
}

void owner_accessor_chain(Owner<NotBorrowable> &owner) {
  NotBorrowable *p = owner.get();
  someFunction();
  NotBorrowable &r = *owner;
  someFunction();
  char &c = owner->at(0);
  someFunction();
  (void)p;
  (void)r;
  (void)c;
}

void not_borrowable(NotBorrowable &n) {
  char &c = n.at(0);
  someFunction();
  n.mutate();
  (void)c;
}

void no_initializer() {
  char *p;
  someFunction();
  (void)p;
}
} // namespace not_a_loan

namespace lifetimebound_edge {
void forwarded_reference(Vector<char> &someVec) {
  Vector<char> &a = forwardRef(someVec);
  // expected-warning@-1{{Local variable 'a' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void forwarded_pointer(Vector<char> &someVec) {
  Vector<char> *a = forwardPtr(someVec);
  // expected-warning@-1{{Local variable 'a' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void self_similar_container(Node &node) {
  Node &child = node.firstChild();
  // expected-warning@-1{{Local variable 'child' is a loan on CanBorrow type 'Node' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  node.appendChild();
  (void)child;
}

void self_similar_container_borrowed(Node &node) {
  Borrow<Node> b(node);
  Node &child = b.get().firstChild();
  someFunction();
  (void)child;
}
} // namespace lifetimebound_edge

namespace guarded {
void through_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char *p = b->data();
  someFunction();
  (void)p;
}

void through_borrow_get(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char &c = b.get()[0];
  someFunction();
  (void)c;
}

void through_borrow_conversion(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char &c = static_cast<Vector<char> &>(b)[0];
  someFunction();
  (void)c;
}

void through_borrow_function(Vector<char> &vec) {
  auto b = borrow(vec);
  char &c = b.get()[0];
  someFunction();
  (void)c;
}

void view_through_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  StringView sv = b.get().data();
  someFunction();
  (void)sv;
}

void borrow_parameter(Borrow<Vector<char>> &b) {
  char &c = b.get()[0];
  someFunction();
  (void)c;
}
} // namespace guarded

namespace no_guardian_exemption {
void borrow_in_enclosing_scope(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  {
    char &c = vec[0];
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
  }
}

void const_reference_parameter(const Vector<char> &vec) {
  const char &c = vec[0];
  // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void const_pointer_parameter(const Vector<char> *vec) {
  const char *p = vec->data();
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void local_container() {
  Vector<char> vec;
  char &c = vec[0];
  // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  vec.append('x');
}
} // namespace no_guardian_exemption

namespace nested_containers {
void loan_on_outer(Vector<Vector<char>> &outer) {
  Vector<char> &inner = outer[0];
  // expected-warning@-1{{Local variable 'inner' is a loan on CanBorrow type 'Vector<Vector<char>>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void borrow_on_outer(Vector<Vector<char>> &outer) {
  Borrow<Vector<Vector<char>>> b(outer);
  Vector<char> &inner = b.get()[0];
  someFunction();
  (void)inner;
}

void borrow_on_inner(Vector<Vector<char>> &outer) {
  Borrow<Vector<Vector<char>>> outerBorrow(outer);
  Vector<char> &inner = outerBorrow.get()[0];
  Borrow<Vector<char>> innerBorrow(inner);
  char &c = innerBorrow.get()[0];
  someFunction();
  (void)c;
}

void alias_on_outer(Vector<Vector<char>> &outer) {
  Vector<Vector<char>> &r = outer;
  someFunction();
  (void)r;
}
} // namespace nested_containers

namespace assignment_sink {
void assign_loan(Vector<char> &vec) {
  char *p = nullptr;
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  p = vec.data();
  someFunction();
}

void assign_guarded(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char *p = nullptr;
  p = b.get().data();
  someFunction();
  (void)p;
}

void assign_alias(Vector<char> &vec) {
  Vector<char> *p = nullptr;
  p = &vec;
  someFunction();
}
} // namespace assignment_sink

namespace conditional_origin {
void one_unguarded_arm(Vector<char> &vec, bool flag) {
  Borrow<Vector<char>> b(vec);
  char *p = flag ? b.get().data() : vec.data();
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void both_arms_guarded(Vector<char> &vec, Vector<char> &other, bool flag) {
  Borrow<Vector<char>> b(vec);
  Borrow<Vector<char>> c(other);
  char *p = flag ? b.get().data() : c.get().data();
  someFunction();
  (void)p;
}
} // namespace conditional_origin

namespace forked_trace {
void call_one_unguarded(Vector<char> &vec, Borrow<Vector<char>> &b) {
  const char *p = pick(b.get().data(), vec.data());
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void call_both_guarded(Borrow<Vector<char>> &b, Borrow<Vector<char>> &c) {
  const char *p = pick(b.get().data(), c.get().data());
  someFunction();
  (void)p;
}

void construct_one_unguarded(Vector<char> &vec, Borrow<Vector<char>> &b) {
  TwoStringViews v(b.get().data(), vec.data());
  // expected-warning@-1{{Local variable 'v' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void construct_both_guarded(Borrow<Vector<char>> &b, Borrow<Vector<char>> &c) {
  TwoStringViews v(b.get().data(), c.get().data());
  someFunction();
  (void)v;
}
} // namespace forked_trace

namespace escape_paths {
void out_parameter_by_reference(Vector<char> &vec, char *&out) {
  out = vec.data();
  // expected-warning@-1{{Parameter 'out' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void outer_scope_local(Vector<char> &vec) {
  char *p = nullptr;
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  {
    p = vec.data();
    someFunction();
  }
  someFunction();
  (void)p;
}

void borrow_outlives_assignment(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char *p = nullptr;
  {
    p = b.get().data();
    someFunction();
  }
  someFunction();
  (void)p;
}

void store_through_out_pointer(Vector<char> &vec, char **out) {
  *out = vec.data();
  // expected-warning@-1{{Parameter 'out' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void store_through_local_pointer(Vector<char> &vec) {
  char *p = nullptr;
  char **pp = &p;
  // expected-warning@-1{{Local variable 'pp' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  *pp = vec.data();
  someFunction();
  (void)p;
}

void store_into_pointer_array(Vector<char> &vec, char *arr[]) {
  arr[0] = vec.data();
  // expected-warning@-1{{Parameter 'arr' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void store_through_double_indirection(Vector<char> &vec, char ***out) {
  **out = vec.data();
  // expected-warning@-1{{Parameter 'out' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void alias_through_out_reference(Vector<char> &vec, Vector<char> *&out) {
  out = &vec;
  someFunction();
}

void alias_through_out_pointer(Vector<char> &vec, Vector<char> **out) {
  *out = &vec;
  someFunction();
}

void guarded_store_through_out_pointer(Vector<char> &vec, char **out) {
  Borrow<Vector<char>> b(vec);
  *out = b.get().data();
  someFunction();
}
} // namespace escape_paths

struct OwningBuffer : CanBorrow {
  char *data() LIFETIME_BOUND;
};
OwningBuffer makeOwningBuffer(const Vector<char> &vec LIFETIME_BOUND);
extern const Vector<char> globalVec;

void owning_temporary_from_global() {
  char *p = makeOwningBuffer(globalVec).data();
  // expected-warning@-1{{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  someFunction();
  (void)p;
}

void owning_temporary_from_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char *p = makeOwningBuffer(b.get()).data();
  // expected-warning@-1{{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  someFunction();
  (void)p;
}

void owning_temporary_from_unguarded(Vector<char> &vec) {
  char *p = makeOwningBuffer(vec).data();
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  // expected-warning@-2{{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  someFunction();
  (void)p;
}

void extended_owning_from_unguarded(Vector<char> &vec) {
  const OwningBuffer &buf = makeOwningBuffer(vec);
  // expected-warning@-1{{Local variable 'buf' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)buf;
}

void extended_owning_from_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  const OwningBuffer &buf = makeOwningBuffer(b.get());
  someFunction();
  (void)buf;
}

namespace known_gaps {
void unannotated_view_constructor(Vector<char> &vec) {
  CharSpan s(vec.data());
  // expected-warning@-1{{Local variable 's' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)s;
}

void unannotated_function_parameter(Vector<char> &vec) {
  CharSpan s = makeSpanUnannotated(vec);
  // expected-warning@-1{{Local variable 's' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)s;
}

void escape_from_trivial_block(Vector<char> &vec) {
  char *p = nullptr;
  {
    p = vec.data();
  }
  someFunction();
  (void)p;
}

void borrow_escapes_via_out_reference(Vector<char> &vec, char *&out) {
  Borrow<Vector<char>> b(vec);
  out = b.get().data();
  someFunction();
}

void borrow_escapes_via_out_pointer(Vector<char> &vec, char **out) {
  Borrow<Vector<char>> b(vec);
  *out = b.get().data();
  someFunction();
}

} // namespace known_gaps

namespace range_for {
void reference_loop(Vector<char> &vec) {
  for (char &c : vec) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void auto_reference_loop(Vector<char> &vec) {
  for (auto &c : vec) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void const_reference_loop(const Vector<char> &vec) {
  for (const char &c : vec) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void nested_loop(Vector<Vector<char>> &outer) {
  for (Vector<char> &inner : outer) {
    // expected-warning@-1{{Local variable 'inner' is a loan on CanBorrow type 'Vector<Vector<char>>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)inner;
  }
}

void local_container_loop() {
  Vector<char> vec;
  for (char &c : vec) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void borrow_get_loop(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  for (char &c : b.get()) {
    someFunction();
    (void)c;
  }
}

void borrow_parameter_loop(Borrow<Vector<char>> &b) {
  for (char &c : b.get()) {
    someFunction();
    (void)c;
  }
}

void value_loop(Vector<char> &vec) {
  for (char c : vec) {
    someFunction();
    (void)c;
  }
}

void array_loop() {
  char arr[4];
  for (char &c : arr) {
    someFunction();
    (void)c;
  }
}

void temporary_range_loop() {
  for (char &c : makeVector()) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void named_temporary_loop() {
  Vector<char> &&r = makeVector();
  for (char &c : r) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void returned_pointer_loop() {
  for (char &c : *getVectorPtr()) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void returned_reference_loop() {
  for (char &c : getVectorRef()) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void manual_iterator_loop(Vector<char> &vec) {
  for (char *it = vec.begin(); it != vec.end(); ++it) {
    // expected-warning@-1{{Local variable 'it' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)*it;
  }
}

void copy_into_local(Vector<char> &vec) {
  char *first = vec.data();
  // expected-warning@-1{{Local variable 'first' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  char *second = first;
  someFunction();
  (void)second;
}
} // namespace range_for

namespace structured_bindings {
struct PointerPair {
  char *first;
  char *second;
};
PointerPair getPair(Vector<char> &vec LIFETIME_BOUND);

void from_lifetimebound_call(Vector<char> &vec) {
  auto [a, b] = getPair(vec);
  // expected-warning@-1{{Local variable 'a' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  // expected-warning@-2{{Local variable 'b' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void through_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  auto [first, second] = getPair(b.get());
  someFunction();
  (void)first;
  (void)second;
}

void per_element_sources(Vector<char> &vec, Vector<char> &other) {
  Borrow<Vector<char>> b(other);
  PointerPair p{b.get().data(), vec.data()};
  auto [guarded, unguarded] = PointerPair{b.get().data(), vec.data()};
  // expected-warning@-1{{Local variable 'unguarded' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)p;
  (void)guarded;
}

void non_view_binding(Vector<char> &vec) {
  struct SizeAndData {
    unsigned size;
    char *data;
  };
  auto [size, data] = SizeAndData{vec.size(), vec.data()};
  // expected-warning@-1{{Local variable 'data' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)size;
}
} // namespace structured_bindings

namespace iterator_reach_back {

Registry<char> makeRegistry();
Registry<char> &namedRegistry();
void use(char);

void interior_destroyed_through_cursor() {
  for (Cursor<char> c : makeRegistry()) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Registry<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    char &name = c.value();
    c.remove();
    use(name);
  }
}

void interior_destroyed_named_container() {
  for (Cursor<char> c : namedRegistry()) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Registry<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    char &name = c.value();
    c.remove();
    use(name);
  }
}

void borrowed_temporary_is_silent() {
  Registry<char> &&r = makeRegistry();
  Borrow<Registry<char>> b(r);
  for (Cursor<char> c : b.get()) {
    char &name = c.value();
    use(name);
  }
}
} // namespace iterator_reach_back

namespace dependent_initializers {

template <typename T> void view_from_dependent_paren_init(T &source) {
  StringView view(source.data());
  // expected-warning@-1{{Local variable 'view' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
}

void instantiate_it(Vector<char> &vec) {
  view_from_dependent_paren_init(vec);
}

template <typename T> void never_instantiated(T &source) {
  StringView view(source.data());
  someFunction();
}

} // namespace dependent_initializers

namespace short_lived_temporaries {

struct Key {
  ~Key();
};
Key makeKey();
unsigned indexFor(const Key &);

void unrelated_temporary_is_still_a_loan(Vector<char> &vec) {
  char &c = vec[indexFor(makeKey())];
  // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)c;
}

void unrelated_temporary_through_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char &c = b.get()[indexFor(makeKey())];
  someFunction();
  (void)c;
}

void borrow_temporary_dies_with_the_statement(Vector<char> &vec) {
  char &c = borrow(vec).get()[0];
  // expected-warning@-1{{temporary bound to local reference 'c' will be destroyed at the end of the full-expression}}
  // expected-warning@-2{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)c;
}

} // namespace short_lived_temporaries
