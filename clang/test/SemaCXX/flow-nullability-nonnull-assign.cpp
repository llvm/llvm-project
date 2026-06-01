// flow-nullability-nonnull-assign.cpp - Tests for assigning null to _Nonnull.
//
// Covers: field default init, member assignment, local variable assignment,
// and dereference after null assignment to _Nonnull.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-nonnull %s -verify

struct Config { int timeout; };

// --- Gap 1: _Nonnull field initialized with nullptr in class body ---

struct FieldInitNull {
  Config* _Nonnull config_ = nullptr; // expected-warning{{initializing nonnull member 'config_'}} expected-note{{remove '_Nonnull'}}
};

struct FieldInitZero {
  Config* _Nonnull config_ = 0; // expected-warning{{initializing nonnull member 'config_'}} expected-note{{remove '_Nonnull'}}
};

struct FieldInitOk {
  Config* config_ = nullptr; // no warning — not _Nonnull
};

struct FieldNoInit {
  Config* _Nonnull config_; // no warning — no initializer
};

// --- Gap 2: assigning nullptr to _Nonnull member inside method ---

struct MemberAssignNull {
  Config* _Nonnull config_;
  MemberAssignNull(Config* _Nonnull c) : config_(c) {}
  void reset() {
    config_ = nullptr; // expected-warning{{assigning nullable pointer to nonnull member 'config_'}} expected-note{{add a null check}}
  }
};

struct MemberAssignOk {
  Config* _Nonnull config_;
  MemberAssignOk(Config* _Nonnull c) : config_(c) {}
  void swap(Config* _Nonnull other) {
    config_ = other; // no warning — nonnull to nonnull
  }
};

// --- Gap 3: assigning nullptr to _Nonnull local variable ---

void local_assign_null() {
  Config c;
  Config* _Nonnull p = &c;
  p = nullptr; // expected-warning{{assigning nullable pointer to nonnull variable 'p'}} expected-note{{add a null check}}
}

void local_assign_ok() {
  Config c1, c2;
  Config* _Nonnull p = &c1;
  p = &c2; // no warning — address-of is nonnull
}

// --- Gap 4: dereference after null assignment to _Nonnull member ---

struct DerefAfterNull {
  Config* _Nonnull config_;
  DerefAfterNull(Config* _Nonnull c) : config_(c) {}
  int bad() {
    config_ = nullptr; // expected-warning{{assigning nullable pointer to nonnull member 'config_'}} expected-note{{add a null check}}
    return config_->timeout; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
};

// Deref after null assign to local
int local_deref_after_null() {
  Config c;
  Config* _Nonnull p = &c;
  p = nullptr; // expected-warning{{assigning nullable pointer to nonnull variable 'p'}} expected-note{{add a null check}}
  return p->timeout; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Gap 5: Constructor argument checking (CXXConstructExpr) ---

struct TakesNonnull {
  int *_Nonnull ptr_;
  TakesNonnull(int *_Nonnull p) : ptr_(p) {}
};

void ctor_nullable_to_nonnull(int *_Nullable p) {
  TakesNonnull t(p); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check}}
}

void ctor_nonnull_ok(int *_Nonnull p) {
  TakesNonnull t(p); // no warning
}

void ctor_narrowed_ok(int *_Nullable p) {
  if (p) {
    TakesNonnull t(p); // no warning — checked
  }
}

void ctor_nullptr() {
  TakesNonnull t(nullptr); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check}}
}

// --- Gap 6: Aggregate InitListExpr checking ---

struct AggNonnull {
  int *_Nonnull p;
  int *_Nonnull q;
};

void aggregate_init_null() {
  int x;
  AggNonnull a1 = {nullptr, &x}; // expected-warning{{assigning nullable pointer to nonnull member}} expected-note{{add a null check}}
}

void aggregate_init_nullable(int *_Nullable p) {
  int x;
  AggNonnull a2 = {p, &x}; // expected-warning{{assigning nullable pointer to nonnull member}} expected-note{{add a null check}}
}

void aggregate_init_ok(int *_Nonnull p) {
  int x;
  AggNonnull a3 = {p, &x}; // no warning
}
