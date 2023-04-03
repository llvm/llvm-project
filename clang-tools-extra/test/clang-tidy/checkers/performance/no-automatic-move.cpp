// RUN: %check_clang_tidy -std=c++11-or-later %s performance-no-automatic-move %t

struct Obj {
  Obj();
  Obj(const Obj &);
  Obj(Obj &&);
  virtual ~Obj();
};

template <typename T>
struct StatusOr {
  StatusOr(const T &);
  StatusOr(T &&);
};

struct NonTemplate {
  NonTemplate(const Obj &);
  NonTemplate(Obj &&);
};

template <typename T>
T Make();

StatusOr<Obj> PositiveStatusOrConstValue() {
  const Obj obj = Make<Obj>();
  return obj;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj' prevents automatic move [performance-no-automatic-move]
}

NonTemplate PositiveNonTemplateConstValue() {
  const Obj obj = Make<Obj>();
  return obj;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj' prevents automatic move [performance-no-automatic-move]
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
StatusOr<Obj> PositiveStatusOrLifetimeExtension() {
  const Obj &obj = Make<Obj>();
  return obj;
}

// Negatives.

StatusOr<Obj> Temporary() {
  return Make<Obj>();
}

StatusOr<Obj> ConstTemporary() {
  return Make<const Obj>();
}

StatusOr<Obj> ConvertingMoveConstructor() {
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

StatusOr<Obj> Ref() {
  Obj &obj = Make<Obj &>();
  return obj;
}

StatusOr<Obj> ConstRef() {
  const Obj &obj = Make<Obj &>();
  return obj;
}

const Obj global;

StatusOr<Obj> Global() {
  return global;
}

struct FromConstRefOnly {
  FromConstRefOnly(const Obj &);
};

FromConstRefOnly FromConstRefOnly() {
  const Obj obj = Make<Obj>();
  return obj;
}
