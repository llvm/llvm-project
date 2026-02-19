// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

struct View;

struct [[gsl::Owner]] MyObj {
  int id;
  MyObj(int i) : id(i) {}
  MyObj() {}
  ~MyObj() {}
  MyObj operator+(MyObj);
  View getView() const [[clang::lifetimebound]];
};

struct [[gsl::Pointer()]] View {
  View(const MyObj &);
  View();
  void use() const;
};

View return_view(View a) {
  // CHECK: :[[@LINE-1]]:18: warning: parameter in intra-TU function should be marked {{\[\[}}clang::lifetimebound]] [-Wlifetime-safety-intra-tu-suggestions]
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:24-[[@LINE-2]]:24}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

MyObj &return_multi(MyObj &a, bool c, MyObj &b) {
  // CHECK-DAG: :[[@LINE-1]]:21: warning: parameter in intra-TU function should be marked
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:29-[[@LINE-2]]:29}:" {{\[\[}}clang::lifetimebound]]"
  // CHECK-DAG: :[[@LINE-3]]:39: warning: parameter in intra-TU function should be marked
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-4]]:47-[[@LINE-4]]:47}:" {{\[\[}}clang::lifetimebound]]"
  if (c)
    return a;
  return b;
}

View return_partial(View a [[clang::lifetimebound]], bool c, View b) {
  // CHECK: :[[@LINE-1]]:62: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:68-[[@LINE-2]]:68}:" {{\[\[}}clang::lifetimebound]]"
  if (c)
    return a;
  return b;
}

View param_with_attr(View a [[maybe_unused]]) {
  // CHECK: :[[@LINE-1]]:22: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:28-[[@LINE-2]]:28}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

View param_default(View a = View()) {
  // CHECK: :[[@LINE-1]]:20: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:35-[[@LINE-2]]:35}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

// FIXME: Iterate over redecls and add [[clang::lifetimebound]]
View multi_decl(View a);
View multi_decl(View a);
View multi_decl(View a) {
  // CHECK: :[[@LINE-1]]:17: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:23-[[@LINE-2]]:23}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

template <typename T>
T *template_identity(T *a) {
  // CHECK: :[[@LINE-1]]:22: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:26-[[@LINE-2]]:26}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

MyObj *instantiate_template() {
  MyObj local;
  return template_identity(&local);
}

struct ViewMember {
  ViewMember(int d) : data(d) {}
  ~ViewMember() {}
  MyObj data;

  View get_view() {
    // CHECK: :[[@LINE-1]]:18: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }

  View get_view_const() const {
    // CHECK: :[[@LINE-1]]:30: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:30-[[@LINE-2]]:30}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }

  const View get_view_const_noexcept() const noexcept {
    // CHECK: :[[@LINE-1]]:54: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:54-[[@LINE-2]]:54}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }
};

struct Base {
  Base() {}
  virtual ~Base() {}
  MyObj data;
  virtual const MyObj &get_virtual() const {
    // CHECK: :[[@LINE-1]]:43: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:43-[[@LINE-2]]:43}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }
};

struct Derived : Base {
  const MyObj &get_virtual() const override {
    // CHECK: :[[@LINE-1]]:35: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:35-[[@LINE-2]]:35}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }
};

struct DerivedFinal : Base {
  const MyObj &get_virtual() const final {
    // CHECK: :[[@LINE-1]]:35: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:35-[[@LINE-2]]:35}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }
};

struct OutOfLine {
  OutOfLine() {}
  ~OutOfLine() {}
  const OutOfLine &get() const;
};
const OutOfLine &OutOfLine::get() const {
  // CHECK: :[[@LINE-1]]:40: warning: implicit this in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:40-[[@LINE-2]]:40}:" {{\[\[}}clang::lifetimebound]]"
  return *this;
}

struct TrailingReturn {
  TrailingReturn() {}
  ~TrailingReturn() {}
  MyObj data;

  auto get_view() -> View {
    // CHECK: :[[@LINE-1]]:18: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }

  auto get_view_const() const -> View {
    // CHECK: :[[@LINE-1]]:30: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:30-[[@LINE-2]]:30}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }

  auto get_ref() const -> const MyObj & {
    // CHECK: :[[@LINE-1]]:23: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:23-[[@LINE-2]]:23}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }
};
