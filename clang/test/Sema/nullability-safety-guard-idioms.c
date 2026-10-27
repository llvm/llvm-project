// C subset of SemaCXX/nullability-safety-guard-idioms.cpp: guard idioms the
// flow-sensitive nullability analysis must understand. FP cases must not warn
// (confirmed false positives); TP cases must keep warning.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 %s -verify=expected,nullable
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c11 %s -verify=expected,nonnull
// RUN: %clang_cc1 -fsyntax-only -std=c11 %s -verify=off
// off-no-diagnostics

typedef _Bool bool;
#define true 1
#define false 0
#define NULL ((void *)0)

struct S {
  int x;
  int *_Nullable next;
  struct S *_Nullable s;
};

struct C {
  int *_Nullable m;
};

int *_Nonnull nonnull(void);
int *_Nullable nullable(void);
int *fallback(void);
extern int gx, gy;
int cond(void);
void takes(int *_Nonnull);

//===----------------------------------------------------------------------===//
// FP: bool/int guards
//===----------------------------------------------------------------------===//

// FP (must not warn after fix)
void b01(int *_Nullable p) {
  bool b = p ? true : false;
  if (b) *p = 1;
}

// FP: one arm a constant false, the other not constant
void b01a(int *_Nullable p) {
  int n = p ? *p : 0;
  if (n) *p = 1;
}

// FP
void b01b(int *_Nullable p) {
  int n = p == NULL ? 0 : *p;
  if (n == 2) *p = 1;
}

// TP: the constant arm is true, so the guard being true proves nothing
void b01c(int *_Nullable p, int c) {
  int n = p ? c : 1;
  if (n) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// FP: guard stored in a struct field
struct Flags {
  int ok;
  int *_Nullable q;
};
void b01d(int *_Nullable p) {
  struct Flags f;
  f.ok = p != NULL;
  if (f.ok) *p = 1;
}

// TP: the struct is assigned as a whole, dropping the field guard
void b01e(int *_Nullable p, struct Flags other) {
  struct Flags f;
  f.ok = p != NULL;
  f = other;
  if (f.ok) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: whole-struct assignment also drops a checked member path
void b01f(int *_Nullable p, struct Flags other) {
  struct Flags f = other;
  f.q = p;
  if (!f.q) return;
  f = other;
  *f.q = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// FP
void b02(int *_Nullable p) {
  int b = p ? false : true;
  if (!b) *p = 1;
}

// FP
void b03(int *_Nullable p) {
  int b;
  b = p != NULL;
  if (b) *p = 1;
}

// FP
void b04(int *_Nullable p, int *_Nullable q) {
  int b = p && q;
  if (b) { *p = 1; *q = 1; }
}

// FP
void b05(int *_Nullable p, int *_Nullable q) {
  int b = p != NULL && q != NULL;
  if (!b) return;
  *p = 1;
  *q = 1;
}

// FP
void b06(int *_Nullable p) {
  int f = p != 0;
  if (f) *p = 1;
}

// FP
void b07(int *_Nullable p) {
  int b;
  if ((b = (p != NULL))) *p = 1;
}

// FP
void b08(struct S *_Nonnull s) {
  int b = s->next != NULL;
  if (b) *s->next = 1;
}

// FP
void b09(int *_Nullable p) {
  int b = p != NULL;
  int c = b;
  if (c) *p = 1;
}

// FP
void b10(int *_Nullable p) {
  bool b = p != NULL;
  bool c = !b;
  if (!c) *p = 1;
}

// FP
void b11(int *_Nullable p) {
  bool b = p != NULL;
  if (b == true) *p = 1;
}

// FP
void b12(int *_Nullable p) {
  bool b = p == NULL;
  if (b == false) *p = 1;
}

// FP
void b14(int *_Nullable p) {
  int b = p != NULL ? true : false;
  if (b) *p = 1;
}

// FP
void b15(int *_Nullable p, int *_Nullable q) {
  int b = !(p && q);
  if (!b) { *p = 1; *q = 1; }
}

// FP
void b16(int *_Nullable p) {
  int f = p ? 1 : 0;
  if (f) *p = 1;
}

// FP
void b17(int *_Nullable p) {
  bool b = p != NULL;
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

int *_Nullable ct01(int c) {
  int *_Nullable p = c ? &gx : (int *)0;
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

// FP: by-value struct member (C stand-in for the C++ reference case)
void m07(struct C c) {
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

// FP: cast-null comparison
void x01(int *_Nullable p) {
  if (p == (int *)0) return;
  *p = 1;
}

// FP
void x02(int *_Nullable p) {
  if (__builtin_expect((long)(p == NULL), 0)) return;
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
  int b = p != NULL;
  p = nullable();
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: || proves nothing about either operand
void tp02(int *_Nullable p, int *_Nullable q) {
  int b = p || q;
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: !(p && q) proves nothing about p
void tp03(int *_Nullable p, int *_Nullable q) {
  int b = p && q;
  if (!b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: flag overwritten after the check
void tp04(int *_Nullable p) {
  int b = p != NULL;
  b = cond();
  if (b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: negated guard
void tp05(int *_Nullable p) {
  bool b = p != NULL;
  if (!b) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: int flag modified after the check
void tp06(int *_Nullable p) {
  int f = p != NULL;
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

// TP: member path reassigned after the flag captured it
void tp12(struct S *_Nonnull s) {
  int b = s->next != NULL;
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

// TP: ternary-derived flag, pointer reassigned afterwards
void tp15(int *_Nullable p) {
  int b = p ? true : false;
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
  bool b = p != NULL;
  if (b == false) *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// FP: an explicit _Nonnull cast asserts non-null at the use
//===----------------------------------------------------------------------===//

// FP
void nc01(int *_Nullable p) { takes((int *_Nonnull)p); }

// FP
void nc02(int *_Nullable p) { *(int *_Nonnull)p = 1; }

// FP
void nc03(struct S *_Nullable s) { ((struct S *_Nonnull)s)->x = 1; }

// FP
void nc04(int *_Nullable p) { ((int *_Nonnull)p)[0] = 1; }

void nc05(int *_Nullable p) {
  takes((int *_Nonnull)(int *_Nullable)p);
}

void nc06(int *_Nonnull p) {
  takes((int *_Nullable)p);
  (void)*(int *_Nullable)p;
}

//===----------------------------------------------------------------------===//
// TP: casts that assert nothing
//===----------------------------------------------------------------------===//

// TP: an unannotated cast still reaches the tracked pointer
void tpc01(int *_Nullable p) {
  *(int *)p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// TP: a _Nullable cast is not an assertion
void tpc02(int *_Nullable p) {
  takes((int *_Nullable)p); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void tpc03(int *_Nullable p) {
  takes((int *_Nullable)(int *_Nonnull)p); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void tpc04(int *_Nullable p) {
  takes((int *_Nonnull)0); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void tpc05(int *_Nonnull p) {
  p = (int *)(int *_Nonnull)0; // expected-warning {{assigning nullable pointer to nonnull variable}} expected-note {{add a null check}}
}
