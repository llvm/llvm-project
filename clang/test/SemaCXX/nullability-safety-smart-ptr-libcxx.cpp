// Smart pointers shaped like libc++ in C++20 mode: shared_ptr declares only
// operator==(const shared_ptr &, nullptr_t), so sp != nullptr and
// nullptr == sp are rewritten comparisons, and operator->, operator* and
// operator bool are members of shared_ptr itself. Most cases take a
// _Nullable parameter so that they mean the same in both default modes; an
// unannotated, unchecked smart pointer is trusted under the nonnull default.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++20 %s -verify=expected,nullable
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++20 %s -verify=expected

namespace std {
using nullptr_t = decltype(nullptr);
template <class T> T &&move(T &) noexcept;
template <class T> class shared_ptr {
public:
  shared_ptr() noexcept;
  shared_ptr(nullptr_t) noexcept;
  shared_ptr(const shared_ptr &) noexcept;
  shared_ptr(shared_ptr &&) noexcept;
  template <class U> shared_ptr(const shared_ptr<U> &) noexcept;
  template <class U> shared_ptr(shared_ptr<U> &&) noexcept;
  ~shared_ptr();
  shared_ptr &operator=(const shared_ptr &) noexcept;
  shared_ptr &operator=(shared_ptr &&) noexcept;
  T *get() const noexcept;
  T &operator*() const noexcept;
  T *operator->() const noexcept;
  explicit operator bool() const noexcept;
  void reset() noexcept;
  void swap(shared_ptr &) noexcept;
};
template <class T>
bool operator==(const shared_ptr<T> &, nullptr_t) noexcept;
template <class T, class... A> shared_ptr<T> make_shared(A &&...);
template <class T, class U>
shared_ptr<T> static_pointer_cast(const shared_ptr<U> &) noexcept;
template <class T, class U>
shared_ptr<T> static_pointer_cast(shared_ptr<U> &&) noexcept;
template <class T, class U>
shared_ptr<T> const_pointer_cast(const shared_ptr<U> &) noexcept;
template <class T, class U>
shared_ptr<T> const_pointer_cast(shared_ptr<U> &&) noexcept;
template <class T, class U>
shared_ptr<T> reinterpret_pointer_cast(const shared_ptr<U> &) noexcept;
template <class T, class U>
shared_ptr<T> dynamic_pointer_cast(const shared_ptr<U> &) noexcept;
template <class T, class U>
shared_ptr<T> dynamic_pointer_cast(shared_ptr<U> &&) noexcept;
} // namespace std

struct S {
  int x;
};

[[noreturn]] void fatal();

//===----------------------------------------------------------------------===//
// S0: a declaration runs once per loop iteration and creates a new smart
// pointer, so a move or reset at the end of one iteration does not reach the
// next iteration's declaration.
//===----------------------------------------------------------------------===//

std::shared_ptr<S> s0_make();
std::shared_ptr<S> _Nullable s0_make_nullable();
void s0_sink(std::shared_ptr<S>);

void s0_moved_at_end(int n) {
  for (int i = 0; i < n; ++i) {
    std::shared_ptr<S> p = s0_make();
    p->x = i; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
    s0_sink(std::move(p));
  }
}

void s0_checked_then_reset(int n) {
  for (int i = 0; i < n; ++i) {
    std::shared_ptr<S> p = s0_make();
    if (!p)
      continue;
    p->x = i;
    p.reset();
  }
}

void s0_nullable_still_warns(int n) {
  for (int i = 0; i < n; ++i) {
    std::shared_ptr<S> p = s0_make_nullable();
    p->x = i; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    p.reset();
  }
}

// The moved-from pointer itself stays tainted until the loop redeclares it.
int s0_used_after_loop_body(int n) {
  std::shared_ptr<S> p = s0_make();
  for (int i = 0; i < n; ++i)
    s0_sink(std::move(p));
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// S4: the rewritten operator can sit anywhere in a chain of !, as in the
// expansions of assertion macros.
//===----------------------------------------------------------------------===//

int s4_not_ne(std::shared_ptr<S> _Nullable p) {
  if (!(p != nullptr))
    return 0;
  return p->x;
}

int s4_assert_trap(std::shared_ptr<S> _Nullable p) {
  do {
    if (!(p != nullptr))
      __builtin_trap();
  } while (0);
  return p->x;
}

int s4_check_expect(std::shared_ptr<S> _Nullable p) {
  while (__builtin_expect(!!(!((p) != nullptr)), 0))
    fatal();
  return p->x;
}

int s4_negated_expect(std::shared_ptr<S> _Nullable p) {
  if (!__builtin_expect(p != nullptr, 1))
    return 0;
  return p->x;
}

int s4_not_not_ne(std::shared_ptr<S> _Nullable p) {
  if (!!(p != nullptr))
    return (*p).x;
  return 0;
}

int s4_not_reversed_ne(std::shared_ptr<S> _Nullable p) {
  if (!(nullptr != p))
    return 0;
  return p->x;
}

int s4_guard(std::shared_ptr<S> _Nullable p) {
  const bool missing = !(p != nullptr);
  if (missing)
    return 0;
  return p->x;
}

int s4_unannotated(std::shared_ptr<S> p) {
  if (!(p != nullptr))
    return 0;
  return p->x;
}

int s4_ne(std::shared_ptr<S> _Nullable p) {
  if (p != nullptr)
    return p->x;
  return 0;
}

int s4_not_eq(std::shared_ptr<S> _Nullable p) {
  if (!(p == nullptr))
    return p->x;
  return 0;
}

int s4_wrong_way(std::shared_ptr<S> _Nullable p) {
  if (!(p != nullptr))
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int s4_guard_wrong_way(std::shared_ptr<S> _Nullable p) {
  const bool missing = !(p != nullptr);
  if (!missing)
    return 0;
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// S2: a smart pointer reached through a reference (a reference parameter, a
// local reference, a reference member) is tracked like one held by value:
// null checks narrow it, and reset(), assignment and std::move through the
// reference drop the narrowing.
//===----------------------------------------------------------------------===//

std::shared_ptr<S> lookup(int id);
std::shared_ptr<S> _Nullable lookup_nullable(int id);
void consume(std::shared_ptr<S>);
void mutate();

struct RefHolder {
  const std::shared_ptr<S> _Nullable &r;
  const std::shared_ptr<S> &u;
};

int s2_eq_cref(const std::shared_ptr<S> _Nullable &p) {
  if (p == nullptr)
    return 0;
  return p->x;
}

int s2_ne_cref(const std::shared_ptr<S> _Nullable &p) {
  if (p != nullptr)
    return (*p).x;
  return 0;
}

int s2_reversed_eq_cref(const std::shared_ptr<S> _Nullable &p) {
  if (nullptr == p)
    return 0;
  return p->x;
}

int s2_or_cref(const std::shared_ptr<S> _Nullable &a,
               const std::shared_ptr<S> _Nullable &b) {
  if (a == nullptr || b == nullptr)
    return 0;
  return a->x + b->x;
}

int s2_eq_local_ref(std::shared_ptr<S> _Nullable q) {
  const auto &p = q;
  if (p == nullptr)
    return 0;
  return p->x + q->x;
}

int s2_eq_ref_member(RefHolder h) {
  if (h.r == nullptr)
    return 0;
  return h.r->x;
}

int s2_unchecked_ref_member(RefHolder h) {
  return h.r->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s2_unannotated_ref_member(RefHolder h) {
  return h.u->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s2_unchecked_cref(const std::shared_ptr<S> &p) {
  return p->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s2_reset_ref(std::shared_ptr<S> &p) {
  if (!p)
    return 0;
  p.reset();
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s2_assign_null_ref(std::shared_ptr<S> &p) {
  if (!p)
    return 0;
  p = nullptr;
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s2_assign_nullable_ref(std::shared_ptr<S> &p) {
  if (!p)
    return 0;
  p = lookup_nullable(1);
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s2_assign_unannotated_ref(std::shared_ptr<S> &p) {
  if (!p)
    return 0;
  p = lookup(1);
  return p->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s2_moved_from_ref(std::shared_ptr<S> &p) {
  if (!p)
    return 0;
  consume(std::move(p));
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s2_move_construct_from_ref(std::shared_ptr<S> &p) {
  if (!p)
    return 0;
  auto q = std::move(p);
  int a = q->x;
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s2_move_assign_between_refs(std::shared_ptr<S> &p, std::shared_ptr<S> &q) {
  if (!q)
    return 0;
  p = std::move(q);
  int a = p->x;
  return a + q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// Each iteration binds the reference anew, so a move at the end of one
// iteration does not reach the next one (S0).
void s2_range_for_ref_moved(std::shared_ptr<S> (&arr)[4]) {
  for (auto &q : arr) {
    q->x = 1; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
    consume(std::move(q));
  }
}

// A class deriving from shared_ptr reaches operator== and reset() through a
// derived-to-base cast of its object.
struct DerivedPtr : std::shared_ptr<S> {};

int s2_derived_eq(DerivedPtr d) {
  if (d == nullptr)
    return 0;
  return d->x;
}

int s2_derived_reset(DerivedPtr d) {
  if (!d)
    return 0;
  d.reset();
  return d->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// Known miss, shared with member paths: a call is not assumed to change what
// a reference refers to, so mutate() resetting p goes unnoticed.
int s2_ref_across_call(std::shared_ptr<S> _Nullable &p) {
  if (!p)
    return 0;
  mutate();
  return p->x;
}

//===----------------------------------------------------------------------===//
// S3c: a guard's facts about a smart pointer are dropped when the pointer
// changes (assignment, reset(), swap(), a move from it), as for a raw
// pointer.
//===----------------------------------------------------------------------===//

S *_Nullable s3c_raw_lookup(int id);

int s3c_assign_nullable(std::shared_ptr<S> _Nullable p) {
  const bool ok = p != nullptr;
  p = lookup_nullable(1);
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_assign_unannotated(std::shared_ptr<S> _Nullable p) {
  const bool ok = p != nullptr;
  p = lookup(1);
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_assign_null(std::shared_ptr<S> p) {
  const bool ok = p != nullptr;
  p = nullptr;
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_reset(std::shared_ptr<S> p) {
  const bool ok = p != nullptr;
  p.reset();
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_moved_from(std::shared_ptr<S> p) {
  const bool ok = p != nullptr;
  consume(std::move(p));
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_move_constructed_from(std::shared_ptr<S> p) {
  const bool ok = p != nullptr;
  std::shared_ptr<S> q = std::move(p);
  (void)q;
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_swapped(std::shared_ptr<S> _Nullable p, std::shared_ptr<S> _Nullable q) {
  const bool ok = p != nullptr;
  p.swap(q);
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_reset_through_ref(std::shared_ptr<S> &p) {
  const bool ok = p != nullptr;
  p.reset();
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

struct SmartHolder {
  std::shared_ptr<S> _Nullable sp;
};

int s3c_member_reset(SmartHolder h) {
  const bool ok = h.sp != nullptr;
  h.sp.reset();
  return ok ? h.sp->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3c_kept(std::shared_ptr<S> _Nullable p) {
  const bool ok = p != nullptr;
  return ok ? p->x : 0;
}

// Assigning a pointer to itself changes nothing.
int s3c_self_assign(std::shared_ptr<S> _Nullable p) {
  const bool ok = p != nullptr;
  p = p;
  return ok ? p->x : 0;
}

int s3c_raw_assign(S *p) {
  const bool ok = p != nullptr;
  p = s3c_raw_lookup(1);
  return ok ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// S1: past a dereference the smart pointer is non-null on that path, so only
// the first dereference on each path warns. Assignment and reset() drop the
// fact again, also through a reference, and a dereference on only some
// incoming paths does not narrow the join.
//===----------------------------------------------------------------------===//

int s1_arrow_then_arrow(std::shared_ptr<S> _Nullable p) {
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  int b = p->x;
  int c = (*p).x;
  return a + b + c;
}

int s1_star_then_arrow(const std::shared_ptr<S> _Nullable &p) {
  int a = (*p).x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + p->x;
}

int s1_unannotated(std::shared_ptr<S> p) {
  int a = p->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  return a + p->x;
}

int s1_after_reset(std::shared_ptr<S> p) {
  p.reset();
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + p->x;
}

int s1_after_move(std::shared_ptr<S> p) {
  consume(std::move(p));
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + p->x;
}

int s1_member(SmartHolder h) {
  int a = h.sp->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + h.sp->x;
}

int s1_reassigned(std::shared_ptr<S> _Nullable p) {
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  p = lookup_nullable(1);
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s1_assigned_null_through_ref(std::shared_ptr<S> _Nullable &p) {
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  p = nullptr;
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s1_reset_through_ref(std::shared_ptr<S> _Nullable &p) {
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  p.reset();
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s1_one_path(std::shared_ptr<S> _Nullable p, bool c) {
  int a = 0;
  if (c)
    a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s1_loop(std::shared_ptr<S> _Nullable p, int n) {
  int a = 0;
  for (int i = 0; i < n; ++i)
    a += p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// A raw pointer is not narrowed by a dereference; each one still warns.
int s1_raw(S *_Nullable p) {
  int a = p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return a + p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// S6: a copy of a non-null smart pointer, and a static / const / reinterpret
// pointer cast of one, is non-null; a copy leaves the source's state alone.
// dynamic_pointer_cast yields null when the runtime check fails, whatever its
// argument, so its result may be null under either default, as for a raw
// dynamic_cast.
//===----------------------------------------------------------------------===//

struct Base {
  virtual ~Base();
  int y;
};
struct Derived : Base {
  int x;
};

int s6_copy_after_check(std::shared_ptr<S> _Nullable p) {
  if (!p)
    return 0;
  auto q = p;
  std::shared_ptr<S> r(p);
  return p->x + q->x + r->x;
}

int s6_copy_assign(std::shared_ptr<S> _Nullable p) {
  if (!p)
    return 0;
  std::shared_ptr<S> q;
  q = p;
  return q->x;
}

int s6_copy_of_made() {
  auto p = std::make_shared<S>();
  std::shared_ptr<S> q = p;
  return q->x;
}

int s6_copy_from_ref(const std::shared_ptr<S> _Nullable &p) {
  if (!p)
    return 0;
  auto q = p;
  return q->x;
}

int s6_converting_copy(std::shared_ptr<Derived> _Nullable d) {
  if (!d)
    return 0;
  std::shared_ptr<Base> b = d;
  return b->y;
}

int s6_copy_is_independent(std::shared_ptr<S> _Nullable p) {
  if (!p)
    return 0;
  auto q = p;
  q.reset();
  return p->x;
}

int s6_static_cast(std::shared_ptr<Base> _Nullable p) {
  if (p == nullptr)
    return 0;
  auto q = std::static_pointer_cast<Derived>(p);
  return q->x;
}

int s6_static_cast_assign(std::shared_ptr<Base> _Nullable p) {
  if (!p)
    return 0;
  std::shared_ptr<Derived> q;
  q = std::static_pointer_cast<Derived>(p);
  return q->x;
}

int s6_const_cast(std::shared_ptr<const S> _Nullable p) {
  if (!p)
    return 0;
  auto q = std::const_pointer_cast<S>(p);
  return q->x;
}

int s6_reinterpret_cast(std::shared_ptr<S> _Nullable p) {
  if (!p)
    return 0;
  auto q = std::reinterpret_pointer_cast<Derived>(p);
  return q->x;
}

int s6_static_cast_of_made() {
  auto q = std::static_pointer_cast<Base>(std::make_shared<Derived>());
  return q->y;
}

// The rvalue overload moves the source into the result.
int s6_static_cast_of_moved(std::shared_ptr<Base> _Nullable p) {
  if (!p)
    return 0;
  auto q = std::static_pointer_cast<Derived>(std::move(p));
  int a = q->x;
  return a + p->y; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s6_static_cast_of_moved_assign(std::shared_ptr<Base> _Nullable p) {
  if (!p)
    return 0;
  std::shared_ptr<Derived> q;
  q = std::static_pointer_cast<Derived>(std::move(p));
  int a = q->x;
  return a + p->y; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s6_copy_unchecked(std::shared_ptr<S> p) {
  auto q = p;
  return q->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s6_static_cast_unchecked(std::shared_ptr<Base> p) {
  auto q = std::static_pointer_cast<Derived>(p);
  return q->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s6_dynamic_cast(std::shared_ptr<Base> _Nullable p) {
  if (!p)
    return 0;
  auto q = std::dynamic_pointer_cast<Derived>(p);
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s6_dynamic_cast_of_nonnull(std::shared_ptr<Base> _Nonnull p) {
  std::shared_ptr<Derived> q;
  q = std::dynamic_pointer_cast<Derived>(p);
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s6_dynamic_cast_to_base(std::shared_ptr<Base> _Nonnull p) {
  std::shared_ptr<Base> q = std::dynamic_pointer_cast<Derived>(p);
  return q->y; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s6_dynamic_cast_checked(std::shared_ptr<Base> _Nonnull p) {
  auto q = std::dynamic_pointer_cast<Derived>(p);
  if (!q)
    return 0;
  return q->x;
}

//===----------------------------------------------------------------------===//
// S3a: a guard built from a ternary with one constant false arm narrows:
// c ? X : false holds like c && X, and c ? false : X like !c && X.
//===----------------------------------------------------------------------===//

int s3a_false_first(std::shared_ptr<S> _Nullable p) {
  const bool valid = p == nullptr ? false : p->x > 0;
  return valid ? p->x : 0;
}

int s3a_false_second(std::shared_ptr<S> _Nullable p) {
  const bool valid = p != nullptr ? p->x > 0 : false;
  return valid ? p->x : 0;
}

int s3a_then_and(std::shared_ptr<S> _Nullable p, bool setup) {
  const bool valid = p == nullptr ? false : p->x > 0;
  const bool use = valid && !setup;
  return use ? p->x : 0;
}

int s3a_nested(std::shared_ptr<S> _Nullable p, std::shared_ptr<S> _Nullable q) {
  const bool both = p ? (q ? q->x > 0 : false) : false;
  if (!both)
    return 0;
  return p->x + q->x;
}

int s3a_arm_is_a_check(std::shared_ptr<S> _Nullable p, std::shared_ptr<S> _Nullable q) {
  const bool both = p != nullptr ? q != nullptr : false;
  return both ? p->x + q->x : 0;
}

int s3a_and_guard(std::shared_ptr<S> _Nullable p) {
  const bool ok = p != nullptr && p->x > 0;
  return ok ? p->x : 0;
}

int s3a_guard_of_guard(std::shared_ptr<S> _Nullable p, bool setup) {
  const bool valid = p != nullptr;
  const bool use = valid && !setup;
  return use ? p->x : 0;
}

int s3a_assigned_guard(std::shared_ptr<S> _Nullable p) {
  bool ok;
  ok = p == nullptr ? false : p->x > 0;
  return ok ? p->x : 0;
}

int s3a_wrong_way(std::shared_ptr<S> _Nullable p) {
  const bool valid = p == nullptr ? false : p->x > 0;
  return valid ? 0 : p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3a_true_arm(std::shared_ptr<S> _Nullable p, bool c) {
  const bool valid = p == nullptr ? true : c;
  return valid ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3a_reset_after(std::shared_ptr<S> _Nullable p) {
  const bool valid = p == nullptr ? false : p->x > 0;
  p.reset();
  return valid ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// S3b: a guard stored in a field (d.hasP = p != nullptr) narrows like one in
// a variable. Writing the field, assigning the whole struct, reassigning the
// pointer the struct is reached through, or changing the tested pointer drops
// it.
//===----------------------------------------------------------------------===//

struct Info {
  bool hasP;
  int count;
  S *raw;
};

int s3b_field_guard(std::shared_ptr<S> _Nullable p) {
  Info d;
  d.hasP = p != nullptr;
  return d.hasP ? p->x : 0;
}

int s3b_field_guard_if(std::shared_ptr<S> _Nullable p) {
  Info d;
  d.hasP = p != nullptr;
  d.count = 1;
  if (!d.hasP)
    return 0;
  return p->x;
}

int s3b_through_pointer(std::shared_ptr<S> _Nullable p, Info *_Nonnull info) {
  info->hasP = p != nullptr;
  if (info->hasP == false)
    return 0;
  return p->x;
}

struct Checker {
  bool ok;
  int check(std::shared_ptr<S> _Nullable p) {
    ok = p != nullptr;
    return ok ? p->x : 0;
  }
};

int s3b_field_overwritten(std::shared_ptr<S> _Nullable p, bool c) {
  Info d;
  d.hasP = p != nullptr;
  d.hasP = c;
  return d.hasP ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3b_field_incremented(std::shared_ptr<S> _Nullable p) {
  Info d;
  d.count = p != nullptr;
  ++d.count;
  return d.count ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3b_struct_overwritten(std::shared_ptr<S> _Nullable p, Info other) {
  Info d;
  d.hasP = p != nullptr;
  d = other;
  return d.hasP ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3b_root_reassigned(std::shared_ptr<S> _Nullable p, Info *_Nonnull a,
                        Info *_Nonnull b) {
  a->hasP = p != nullptr;
  a = b;
  return a->hasP ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3b_pointer_reset(std::shared_ptr<S> _Nullable p) {
  Info d;
  d.hasP = p != nullptr;
  p.reset();
  return d.hasP ? p->x : 0; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s3b_wrong_way(std::shared_ptr<S> _Nullable p) {
  Info d;
  d.hasP = p != nullptr;
  return d.hasP ? 0 : p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// Assigning the whole struct also drops narrowing of paths under it.
int s3b_struct_overwritten_member(Info other) {
  Info d = other;
  if (!d.raw)
    return 0;
  d = other;
  return d.raw->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// S9: a smart pointer store classifies the stored value as a raw pointer
// store does. A copy or move carries the source's nullability as well as its
// narrowing, and a value that is neither provably non-null nor nullable
// leaves the target to the default: under the nonnull default, a member
// path assigned an unannotated value is trusted, as a local is.
//===----------------------------------------------------------------------===//

int s9_copy_of_nullable(std::shared_ptr<S> _Nullable p) {
  std::shared_ptr<S> q = p;
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s9_copy_of_reset(std::shared_ptr<S> p) {
  p.reset();
  std::shared_ptr<S> q(p);
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s9_copy_assign_of_nullable(std::shared_ptr<S> _Nullable p) {
  std::shared_ptr<S> q = lookup(0);
  q = p;
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s9_move_of_nullable(std::shared_ptr<S> _Nullable p) {
  std::shared_ptr<S> q = std::move(p);
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s9_move_assign_of_moved_from(std::shared_ptr<S> p, std::shared_ptr<S> q) {
  consume(std::move(p));
  q = std::move(p);
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s9_copy_of_checked(std::shared_ptr<S> _Nullable p) {
  if (!p)
    return 0;
  std::shared_ptr<S> q = p;
  return q->x;
}

struct Owner {
  std::shared_ptr<S> m;

  int assign_unannotated() {
    m = lookup(1);
    return m->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  }

  int move_unannotated(std::shared_ptr<S> p) {
    m = std::move(p);
    return m->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  }

  int assign_nullable() {
    m = lookup_nullable(1);
    return m->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }

  int move_nullable(std::shared_ptr<S> _Nullable p) {
    m = std::move(p);
    return m->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }

  int copy_reset(std::shared_ptr<S> p) {
    p.reset();
    m = p;
    return m->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }
};

int s9_member_of_variable(Owner &o) {
  o.m = lookup(1);
  return o.m->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}
// S7a: a member reached through a smart pointer's -> or * (p->child,
// (*p).child) is a path rooted at p, as p->child is for a raw pointer p: it
// is checked, narrowed by a null check, and forgotten when p changes. This
// covers raw pointer members too (p->raw->x).
//===----------------------------------------------------------------------===//

struct Node {
  int x;
  std::shared_ptr<Node> _Nullable child;
  std::shared_ptr<Node> next;
  Node *_Nullable raw;
  Node *plain;
};

int s7a_chained(std::shared_ptr<Node> _Nullable p) {
  if (!p)
    return 0;
  return p->child->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7a_chained_star(std::shared_ptr<Node> _Nullable p) {
  if (!p)
    return 0;
  return (*p->child).x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7a_chained_through_star(std::shared_ptr<Node> _Nullable p) {
  if (!p)
    return 0;
  return (*p).child->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7a_chained_checked(std::shared_ptr<Node> _Nullable p) {
  if (!p || !p->child)
    return 0;
  return p->child->x;
}

// (*p).child and p->child are the same path.
int s7a_chained_compared(std::shared_ptr<Node> _Nullable p) {
  if (p == nullptr || p->child == nullptr)
    return 0;
  return (*p).child->x;
}

int s7a_unannotated_member(std::shared_ptr<Node> _Nullable p) {
  if (!p)
    return 0;
  return p->next->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s7a_raw_member(std::shared_ptr<Node> _Nullable p) {
  if (!p)
    return 0;
  return p->raw->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7a_raw_member_checked(std::shared_ptr<Node> _Nullable p) {
  if (!p || !p->raw)
    return 0;
  return p->raw->x;
}

int s7a_unannotated_raw_member(std::shared_ptr<Node> _Nullable p) {
  if (!p)
    return 0;
  return p->plain->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int s7a_raw_member_assigned(std::shared_ptr<Node> _Nullable p,
                            Node *_Nonnull n) {
  if (!p)
    return 0;
  p->raw = n;
  int a = p->raw->x;
  p->raw = nullptr;
  return a + p->raw->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7a_root_reassigned(std::shared_ptr<Node> _Nullable p,
                        std::shared_ptr<Node> _Nullable o) {
  if (!p || !p->child)
    return 0;
  p = o;
  if (!p)
    return 0;
  return p->child->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// Assigning p->child forgets what was known below the old child.
int s7a_member_reassigned(std::shared_ptr<Node> _Nullable p) {
  if (!p || !p->child || !p->child->child || !p->child->child->child)
    return 0;
  p->child = p->child->child;
  int a = p->child->x;
  return a + p->child->child->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7a_reference_root(std::shared_ptr<Node> _Nullable q) {
  auto &p = q;
  if (!q || !q->child)
    return 0;
  return p->child->x;
}

struct Tree {
  std::shared_ptr<Node> _Nullable root;
  int depth() {
    if (!root || !root->child)
      return 0;
    return root->child->x;
  }
};

//===----------------------------------------------------------------------===//
// S7b: a smart pointer returned by a call or by operator[] has no identity to
// narrow, so a dereference of it warns only when it may be null by contract:
// the callee declares a _Nullable return, or it is std::dynamic_pointer_cast.
// An unannotated return is not reported (a local bound to it is, under the
// nullable default).
//===----------------------------------------------------------------------===//

struct Map {
  std::shared_ptr<S> &operator[](int);
};
struct NullableMap {
  std::shared_ptr<S> _Nullable &operator[](int);
};
struct Registry {
  std::shared_ptr<S> _Nullable get(int id) const;
};
std::shared_ptr<S> _Nonnull lookup_nonnull(int id);
template <class T> std::shared_ptr<T> _Nullable find_nullable(int id);

int s7b_nullable_call() {
  return lookup_nullable(1)->x; // expected-warning {{dereference of nullable pointer 'std::shared_ptr<S> _Nullable'}} expected-note {{add a null check}}
}

int s7b_nullable_call_star() {
  return (*lookup_nullable(1)).x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7b_nullable_subscript(NullableMap &m) {
  return m[1]->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7b_nullable_method(const Registry &r) {
  return r.get(1)->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7b_nullable_template() {
  return find_nullable<S>(1)->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7b_dynamic_cast(std::shared_ptr<Base> _Nonnull p) {
  return std::dynamic_pointer_cast<Derived>(p)->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int s7b_unannotated_call() {
  return lookup(1)->x;
}

int s7b_unannotated_subscript(Map &m) {
  return m[1]->x;
}

int s7b_nonnull_call() {
  return lookup_nonnull(1)->x;
}

int s7b_static_cast(std::shared_ptr<Base> _Nonnull p) {
  return std::static_pointer_cast<Derived>(p)->x;
}

int s7b_named_local() {
  auto q = lookup_nullable(1);
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
