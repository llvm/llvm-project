// RUN: %check_clang_tidy -std=c++11-or-later %s performance-no-automatic-move %t -- \
// RUN:   -config="{CheckOptions: {performance-no-automatic-move.AllowedTypes: '::Obj'}}"

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

template <typename T>
T Make();

NonTemplate PositiveNonTemplate() {
  const Obj obj = Make<Obj>();
  return obj; // No warning because Obj is allowed.
}

struct Other {
  Other();
  Other(const Other &);
  Other(Other &&);
  virtual ~Other();
};

struct OtherNonTemplate {
  OtherNonTemplate(const Other &);
  OtherNonTemplate(Other &&);
};

OtherNonTemplate PositiveOtherNonTemplate() {
  const Other obj = Make<Other>();
  return obj;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: constness of 'obj' prevents automatic move [performance-no-automatic-move]
}
