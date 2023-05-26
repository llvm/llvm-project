// RUN: %check_clang_tidy -std=c++11-or-later %s performance-no-automatic-move %t

struct Obj {
  Obj();
  Obj(const Obj &);
  Obj(Obj &&);
  virtual ~Obj();
};

struct NonTemplate {
  NonTemplate(const Obj &);
  NonTemplate(Obj &&);
};

template <typename T> struct TemplateCtorPair {
  TemplateCtorPair(const T &);
  TemplateCtorPair(T &&value);
};

template <typename T> struct UrefCtor {
  template <class U = T> UrefCtor(U &&value);
};

template <typename T>
T Make();

NonTemplate PositiveNonTemplate() {
  const Obj obj = Make<Obj>();
  return obj; // selects `NonTemplate(const Obj&)`
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj' prevents
  // automatic move [performance-no-automatic-move]
}

TemplateCtorPair<Obj> PositiveTemplateCtorPair() {
  const Obj obj = Make<Obj>();
  return obj; // selects `TemplateCtorPair(const T&)`
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj' prevents
  // automatic move [performance-no-automatic-move]
}

UrefCtor<Obj> PositiveUrefCtor() {
  const Obj obj = Make<Obj>();
  return obj; // selects `UrefCtor(const T&&)`
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj' prevents
  // automatic move [performance-no-automatic-move]
}

Obj PositiveCantNrvo(bool b) {
  const Obj obj1;
  const Obj obj2;
  if (b) {
    return obj1;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: constness of 'obj1' prevents automatic move [performance-no-automatic-move]
  }
  return obj2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj2' prevents automatic move [performance-no-automatic-move]
}

// FIXME: Ideally we would warn here too.
NonTemplate PositiveNonTemplateLifetimeExtension() {
  const Obj &obj = Make<Obj>();
  return obj;
}

// FIXME: Ideally we would warn here too.
UrefCtor<Obj> PositiveUrefCtorLifetimeExtension() {
  const Obj &obj = Make<Obj>();
  return obj;
}

// Negatives.

UrefCtor<Obj> Temporary() { return Make<Obj>(); }

UrefCtor<Obj> ConstTemporary() { return Make<const Obj>(); }

UrefCtor<Obj> ConvertingMoveConstructor() {
  Obj obj = Make<Obj>();
  return obj;
}

Obj ConstNrvo() {
  const Obj obj = Make<Obj>();
  return obj;
}

Obj NotNrvo(bool b) {
  Obj obj1;
  Obj obj2;
  if (b) {
    return obj1;
  }
  return obj2;
}

UrefCtor<Obj> Ref() {
  Obj &obj = Make<Obj &>();
  return obj;
}

UrefCtor<Obj> ConstRef() {
  const Obj &obj = Make<Obj &>();
  return obj;
}

const Obj global;

UrefCtor<Obj> Global() { return global; }

struct FromConstRefOnly {
  FromConstRefOnly(const Obj &);
};

FromConstRefOnly FromConstRefOnly() {
  const Obj obj = Make<Obj>();
  return obj;
}
