// Guard idioms the flow-sensitive nullability analysis must understand.
// FP cases (must not warn) capture confirmed false positives: a null check
// stored in a bool/int, a ternary whose selected arm is guarded, an alias of
// a checked member path, pointer arithmetic after a guard, and self-assign.
// TP cases (must warn) guard against over-suppression once the FPs are fixed.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c++17 %s -verify=expected,nullable
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify=expected,nonnull
// RUN: %clang_cc1 -fsyntax-only -std=c++17 %s -verify=off
// off-no-diagnostics

struct S {
  int x;
  int *_Nullable next;
  struct S *_Nullable s;
};

struct Base {
  virtual ~Base() = default;
};
struct Derived : Base {
  int x;
};

int *_Nonnull nonnull(void);
int *_Nullable nullable(void);
int *fallback(void);
extern int gx, gy;
int cond(void);
void takes(int *_Nonnull);
void takesDerived(Derived *_Nonnull);

//===----------------------------------------------------------------------===//
// FP: bool/int guards
//===----------------------------------------------------------------------===//

// FP (must not warn after fix)
void b01(int *_Nullable p) {
  bool b = p ? true : false;
  if (b) *p = 1;
}

// FP
void b02(int *_Nullable p) {
  bool b = p ? false : true;
  if (!b) *p = 1;
}

// FP
void b03(int *_Nullable p) {
  bool b;
  b = p != nullptr;
  if (b) *p = 1;
}

// FP
void b04(int *_Nullable p, int *_Nullable q) {
  bool b = p && q;
  if (b) { *p = 1; *q = 1; }
}

// FP
void b05(int *_Nullable p, int *_Nullable q) {
  bool b = p != nullptr && q != nullptr;
  if (!b) return;
  *p = 1;
  *q = 1;
}

// FP
void b06(int *_Nullable p) {
  int f = p != nullptr;
  if (f) *p = 1;
}

// FP
void b07(int *_Nullable p) {
  bool b;
  if ((b = (p != nullptr))) *p = 1;
}

// FP
void b08(struct S *_Nonnull s) {
  bool b = s->next != nullptr;
  if (b) *s->next = 1;
}

// FP
void b09(int *_Nullable p) {
  bool b = p != nullptr;
  bool c = b;
  if (c) *p = 1;
}

// FP
void b10(int *_Nullable p) {
  bool b = p != nullptr;
  bool c = !b;
  if (!c) *p = 1;
}

// FP
void b11(int *_Nullable p) {
  bool b = p != nullptr;
  if (b == true) *p = 1;
}

// FP
void b12(int *_Nullable p) {
  bool b = p == nullptr;
  if (b == false) *p = 1;
}

// FP
void b13(int *_Nullable p) {
  bool b{p != nullptr};
  if (b) *p = 1;
}

// FP
void b14(int *_Nullable p) {
  bool b = p != nullptr ? true : false;
  if (b) *p = 1;
}

// FP
void b15(int *_Nullable p, int *_Nullable q) {
  bool b = !(p && q);
  if (!b) { *p = 1; *q = 1; }
}

// FP
void b16(int *_Nullable p) {
  int f = p ? 1 : 0;
  if (f) *p = 1;
}

// FP
void b17(int *_Nullable p) {
  bool b = p != nullptr;
  if (b != false) *p = 1;
}

//===----------------------------------------------------------------------===//
// FP: ternary self-guard
//===----------------------------------------------------------------------===//

// FP
void t01(int *_Nullable p) {
  int *r = p ? p : nonnull();
  *r = 1;
}

// FP
void t02(int *_Nullable p) {
  *(p ? p : &gx) = 1;
}

// FP
int *_Nonnull t03(int *_Nullable p) {
  return p ? p : &gx;
}

// FP
void t04(int *_Nullable p) {
  takes(p ? p : &gx);
}

// FP
void t05(int *_Nullable p) {
  int *r = p ?: nonnull();
  *r = 1;
}

// FP
int *_Nonnull t06(int *_Nullable p) {
  takes(p ?: &gx);
  return p ?: &gx;
}

// FP
void t07(int *_Nullable p) {
  *(p ?: &gx) = 1;
}

// FP
void t08(int *_Nullable p) {
  int *_Nonnull r = p ? p : &gx;
  *r = 1;
}

// FP: neither RUN line passes -fnullability-default=nullable, so an
// unannotated fallback() return is not nullable in either mode (the plain
// `int *r = fallback(); *r` is silent too). The old warning here came only
// from the ternary's merged type inheriting _Nullable from the guarded arm.
void t09(int *_Nullable p) {
  int *r = p ? p : fallback();
  *r = 1;
}

//===----------------------------------------------------------------------===//
// FP: pointer correlated with a condition variable by a ternary init.
// p = c ? &gx : nullptr makes p non-null exactly when c holds, so a later
// if (c) narrows p. The inverse of the bool-guard cases above.
//===----------------------------------------------------------------------===//

// FP
int *_Nullable ct01(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  if (c) *p = 1;
  return p;
}

// FP: arms swapped, so the false edge is the non-null one
int *_Nullable ct02(bool c) {
  int *_Nullable p = c ? nullptr : &gx;
  if (!c) *p = 1;
  return p;
}

// FP: negation inside the ternary condition
int *_Nullable ct03(bool c) {
  int *_Nullable p = !c ? &gx : nullptr;
  if (!c) *p = 1;
  return p;
}

// FP: the guard is copied before the check
int *_Nullable ct04(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  bool d = c;
  if (d) *p = 1;
  return p;
}

// FP: guard is one conjunct of a larger condition
int *_Nullable ct05(bool c, int *_Nullable q) {
  int *_Nullable p = c ? &gx : nullptr;
  if (c && q) *p = 1;
  return p;
}

// FP: int flag rather than bool, the C idiom
int *_Nullable ct06(int flag) {
  int *_Nullable p = flag ? &gx : nullptr;
  if (flag) *p = 1;
  return p;
}

// FP: the non-null arm is a _Nonnull parameter rather than an address-of
int *_Nullable ct07(bool c, int *_Nonnull q) {
  int *_Nullable p = c ? q : nullptr;
  if (c) *p = 1;
  return p;
}

// FP: one condition variable correlating two pointers
int *_Nullable ct08(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  int *_Nullable q = c ? &gy : nullptr;
  if (c) *p = *q;
  return p;
}

// FP: assignment rather than initialization
int *_Nullable ct09(bool c) {
  int *_Nullable p = nullptr;
  p = c ? &gx : nullptr;
  if (c) *p = 1;
  return p;
}

//===----------------------------------------------------------------------===//
// FP: alias of a checked member path
//===----------------------------------------------------------------------===//

// FP
void m01(struct S *_Nonnull s) {
  if (s->next) {
    int *r = s->next;
    *r = 1;
  }
}

// FP
void m02(struct S *_Nonnull s) {
  if (s->next) {
    auto *r = s->next;
    *r = 1;
  }
}

// FP
void m03(struct S *_Nonnull s) {
  int *r = s->next;
  if (r) *s->next = 1;
}

// FP
void m04(struct S *_Nonnull s) {
  int *r = s->next;
  if (s->next) *r = 1;
}

// FP
void m05(struct S *_Nonnull s) {
  if (!s->next) return;
  int *r = s->next;
  *r = 1;
}

// FP
struct C {
  int *_Nullable m;
  void m06() {
    if (!m) return;
    int *r = m;
    *r = 1;
  }
};

// FP
void m07(struct C &c) {
  if (!c.m) return;
  int *r = c.m;
  *r = 1;
}

//===----------------------------------------------------------------------===//
// FP: pointer arithmetic after a guard
//===----------------------------------------------------------------------===//

// FP
void a01(int *_Nullable p) {
  if (p) { *p++ = 1; }
}

// FP
void a02(int *_Nullable p) {
  if (p) { p += 1; *p = 1; }
}

// FP
void a03(int *_Nullable p) {
  if (p) { *(p + 1) = 1; }
}

// FP
void a04(int *_Nullable p) {
  if (p) { *++p = 1; }
}

// FP
void a05(int *_Nullable p) {
  if (p) { p -= 1; *p = 1; }
}

//===----------------------------------------------------------------------===//
// FP: misc
//===----------------------------------------------------------------------===//

// FP
void x02(int *_Nullable p) {
  if (__builtin_expect((long)(p == nullptr), 0)) return;
  *p = 1;
}

// FP
void x03(int *_Nullable p) {
  if (!p) return;
  p = p;
  *p = 1;
}

// FP
void x04(int *_Nullable p) {
  if (!p) return;
  p = (int *)p;
  *p = 1;
}

//===----------------------------------------------------------------------===//
// TP: must keep warning
//===----------------------------------------------------------------------===//

// TP: reassignment invalidates the stored guard
void tp01(int *_Nullable p) {
  bool b = p != nullptr;
  p = nullable();
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: || proves nothing about either operand
void tp02(int *_Nullable p, int *_Nullable q) {
  bool b = p || q;
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: !(p && q) proves nothing about p
void tp03(int *_Nullable p, int *_Nullable q) {
  bool b = p && q;
  if (!b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: bool overwritten after the check
void tp04(int *_Nullable p) {
  bool b = p != nullptr;
  b = cond();
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: negated guard
void tp05(int *_Nullable p) {
  bool b = p != nullptr;
  if (!b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: int flag modified after the check
void tp06(int *_Nullable p) {
  int f = p != nullptr;
  f++;
  if (f) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: false arm nullable
void tp07(int *_Nullable p) {
  int *r = p ? p : nullable();
  *r = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: true arm is not the condition
void tp08(int *_Nullable p) {
  int *r = p ? nullable() : &gx;
  *r = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: GNU ?: with nullable fallback
void tp09(int *_Nullable p) {
  int *r = p ?: nullable();
  *r = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: member path reassigned after copy; later check on the path says nothing about r
void tp10(struct S *_Nonnull s) {
  int *r = s->next;
  s->next = nullable();
  if (s->next) *r = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: alias reassigned after copy
void tp11(struct S *_Nonnull s) {
  int *r = s->next;
  r = nullable();
  if (s->next) *r = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: member path reassigned after the bool captured it
void tp12(struct S *_Nonnull s) {
  bool b = s->next != nullptr;
  s->next = nullable();
  if (b) *s->next = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: unguarded post-increment deref (both the arithmetic and the deref warn)
void tp13(int *_Nullable p) {
  *p++ = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check before dereferencing}} expected-warning {{pointer arithmetic on nullable pointer}} expected-note {{add a null check before performing arithmetic}}
}

// TP: unguarded compound-assign arithmetic
void tp14(int *_Nullable p) {
  p += 1; // expected-warning {{pointer arithmetic on nullable pointer}} expected-note {{if this pointer cannot be null}}
}

// TP: ternary-derived bool, pointer reassigned afterwards
void tp15(int *_Nullable p) {
  bool b = p ? true : false;
  p = nullable();
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: alias checked, but the member path is reassigned before the deref
void tp16(struct S *_Nonnull s) {
  int *r = s->next;
  if (r) {
    s->next = nullable();
    *s->next = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }
}

// TP: b == false is the negated guard
void tp17(int *_Nullable p) {
  bool b = p != nullptr;
  if (b == false) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: correlated ternary, but the deref is not guarded at all
int *_Nullable tp18(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: correlated ternary, pointer reassigned before the check
int *_Nullable tp19(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  p = nullptr;
  if (c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: correlated ternary, condition variable reassigned before the check
int *_Nullable tp20(bool c, bool b) {
  int *_Nullable p = c ? &gx : nullptr;
  c = b;
  if (c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: correlated ternary checked in the wrong direction
int *_Nullable tp21(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  if (!c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: correlated ternary, deref on the else edge
int *_Nullable tp22(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  if (c) {} else *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: the selected arm is itself nullable, so c proves nothing
int *_Nullable tp23(bool c, int *_Nullable q) {
  int *_Nullable p = c ? q : nullptr;
  if (c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: only one predecessor carries the correlation, so the join drops it
int *_Nullable tp24(bool a, bool c) {
  int *_Nullable p = nullptr;
  if (a)
    p = c ? &gx : nullptr;
  else
    p = nullptr;
  if (c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

// TP: a different condition variable is checked
int *_Nullable tp25(bool c, bool d) {
  int *_Nullable p = c ? &gx : nullptr;
  if (d) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

int *_Nullable tp26(bool c) {
  int *_Nullable p = c ? &gx : (c = true, nullptr);
  if (c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

int *_Nullable tp27(bool c) {
  int *_Nullable p = c ? &gx : nullptr;
  bool *pc = &c;
  *pc = true;
  if (c) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return p;
}

//===----------------------------------------------------------------------===//
// FP: an explicit _Nonnull cast asserts non-null at the use
//===----------------------------------------------------------------------===//

// FP
void nc01(int *_Nullable p) { takes((int *_Nonnull)p); }

// FP
void nc02(int *_Nullable p) { takes(static_cast<int *_Nonnull>(p)); }

// FP
void nc03(int *_Nullable p) { *(int *_Nonnull)p = 1; }

// FP
void nc04(S *_Nullable s) { ((S *_Nonnull)s)->x = 1; }

// FP
void nc05(int *_Nullable p) { ((int *_Nonnull)p)[0] = 1; }

// FP: an outer unannotated cast does not undo the assertion
void nc06(int *_Nullable p) { *(int *)(int *_Nonnull)p = 1; }

// FP
int *_Nonnull nc07(int *_Nullable p) {
  int *_Nonnull q = (int *_Nonnull)p;
  return q;
}

// FP
int *_Nonnull nc08(int *_Nullable p) { return (int *_Nonnull)p; }

void nc09(Base *_Nullable p) {
  takesDerived(dynamic_cast<Derived *_Nonnull>(p));
}

Derived *_Nonnull nc10(Base *_Nullable p) {
  return dynamic_cast<Derived *_Nonnull>(p);
}

void nc11(Base *_Nullable p) {
  (void)*dynamic_cast<Derived *_Nonnull>(p);
}

void nc12(int *_Nullable p) {
  takes((int *_Nonnull)(int *_Nullable)p);
}

void nc13(int *_Nonnull p) {
  takes((int *_Nullable)p);
  (void)*(int *_Nullable)p;
}

//===----------------------------------------------------------------------===//
// TP: casts that assert nothing
//===----------------------------------------------------------------------===//

// TP: an unannotated cast still reaches the tracked pointer
void tp28(int *_Nullable p) {
  *(int *)p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: a _Nullable cast is not an assertion
void tp29(int *_Nullable p) {
  takes((int *_Nullable)p); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void tp30(int *_Nullable p) {
  takes((int *_Nullable)(int *_Nonnull)p); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void tp31(int *_Nullable) {
  takes((int *_Nonnull)nullptr); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void tp32(int *_Nonnull p) {
  p = (int *)(int *_Nonnull)nullptr; // expected-warning {{assigning nullable pointer to nonnull variable}} expected-note {{add a null check}}
}

void tp33(Base *_Nonnull p) {
  takesDerived((Derived *_Nullable)dynamic_cast<Derived *_Nonnull>(p)); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}
