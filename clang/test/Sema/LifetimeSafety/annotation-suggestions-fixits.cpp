// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wlifetime-safety-annotation-placement -Wno-dangling \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wlifetime-safety-annotation-placement -Wno-dangling \
// RUN:   -DLIFETIMEBOUND_MACRO=[[clang::lifetimebound]] \
// RUN:   -lifetime-safety-lifetimebound-macro=LIFETIMEBOUND_MACRO \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACRO
// RUN: cp %s %t.cpp
// RUN: %clang_cc1 -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling -fixit %t.cpp
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Werror=lifetime-safety-suggestions -Wno-dangling %t.cpp
// RUN: cp %s %t.bad-macro.cpp
// RUN: %clang_cc1 -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   -lifetime-safety-lifetimebound-macro=BAD_LIFETIMEBOUND_MACRO \
// RUN:   -fixit %t.bad-macro.cpp
// RUN: not %clang_cc1 -fsyntax-only -std=c++17 %t.bad-macro.cpp 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-BAD-MACRO

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
  // CHECK: :[[@LINE-1]]:24: warning: parameter in intra-TU function should be marked {{\[\[}}clang::lifetimebound]] [-Wlifetime-safety-intra-tu-suggestions]
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:24-[[@LINE-2]]:24}:" {{\[\[}}clang::lifetimebound]]"
  // CHECK-MACRO: :[[@LINE-3]]:24: warning: parameter in intra-TU function should be marked
  // CHECK-MACRO: fix-it:"{{.*}}":{[[@LINE-4]]:24-[[@LINE-4]]:24}:" LIFETIMEBOUND_MACRO"
  // CHECK-BAD-MACRO: :[[@LINE-5]]:25: error: expected ')'
  // CHECK-BAD-MACRO: BAD_LIFETIMEBOUND_MACRO
  return a;
}

MyObj &return_multi(MyObj &a, bool c, MyObj &b) {
  // CHECK-DAG: :[[@LINE-1]]:29: warning: parameter in intra-TU function should be marked
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:29-[[@LINE-2]]:29}:" {{\[\[}}clang::lifetimebound]]"
  // CHECK-DAG: :[[@LINE-3]]:47: warning: parameter in intra-TU function should be marked
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-4]]:47-[[@LINE-4]]:47}:" {{\[\[}}clang::lifetimebound]]"
  if (c)
    return a;
  return b;
}

View return_partial(View a [[clang::lifetimebound]], bool c, View b) {
  // CHECK: :[[@LINE-1]]:68: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:68-[[@LINE-2]]:68}:" {{\[\[}}clang::lifetimebound]]"
  if (c)
    return a;
  return b;
}

View param_with_attr(View a [[maybe_unused]]) {
  // CHECK: :[[@LINE-1]]:28: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:28-[[@LINE-2]]:28}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

View param_default(View a = View()) {
  // CHECK: :[[@LINE-1]]:26: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:26-[[@LINE-2]]:26}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

int *arr_default(int a[2] = nullptr) {
  // CHECK: :[[@LINE-1]]:23: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:23-[[@LINE-2]]:23}:" {{\[\[}}clang::lifetimebound]]"
  return a;
}

View multi_decl(View a);
// CHECK: :[[@LINE-1]]:23: warning: parameter in intra-TU function should be marked
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:23-[[@LINE-2]]:23}:" {{\[\[}}clang::lifetimebound]]"
View multi_decl(View a);
View multi_decl(View a) {
  return a;
}

template <typename T>
T *template_identity(T *a) {
  // CHECK: :[[@LINE-1]]:26: warning: parameter in intra-TU function should be marked
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
    // CHECK-BAD-MACRO: :[[@LINE-3]]:18: error: expected ';' at end of declaration list
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
  // CHECK: :[[@LINE-1]]:31: warning: implicit this in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:31-[[@LINE-2]]:31}:" {{\[\[}}clang::lifetimebound]]"
};
const OutOfLine &OutOfLine::get() const {
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

#define GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))

View return_view_with_gnu_macro(View a) {
  // CHECK: :[[@LINE-1]]:39: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:39-[[@LINE-2]]:39}:" GNU_LIFETIMEBOUND_MACRO"
  return a;
}

struct OnlyGNUMember {
  MyObj data;

  View get_view() {
    // CHECK: :[[@LINE-1]]:18: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:" {{\[\[}}clang::lifetimebound]]"
    return data;
  }
};

#define LIFETIMEBOUND_MACRO [[clang::lifetimebound]]
#define MY_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

View unnamed_macro(View);
// CHECK: :[[@LINE-1]]:20: warning: parameter in intra-TU function should be marked
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:20-[[@LINE-2]]:20}:"MY_LIFETIMEBOUND_MACRO "
View unnamed_macro(View a) {
  return a;
}

View return_view_with_macro(View a) {
  // CHECK: :[[@LINE-1]]:35: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:35-[[@LINE-2]]:35}:" MY_LIFETIMEBOUND_MACRO"
  return a;
}

#define FIRST_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]
#define SECOND_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

View return_view_with_latest_macro(View a) {
  // CHECK: :[[@LINE-1]]:42: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:42-[[@LINE-2]]:42}:" SECOND_LIFETIMEBOUND_MACRO"
  // CHECK-MACRO: :[[@LINE-3]]:42: warning: parameter in intra-TU function should be marked
  // CHECK-MACRO: fix-it:"{{.*}}":{[[@LINE-4]]:42-[[@LINE-4]]:42}:" LIFETIMEBOUND_MACRO"
  return a;
}

#define REDEFINED_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

View return_view_with_redefined_macro(View a) {
  // CHECK: :[[@LINE-1]]:45: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:45-[[@LINE-2]]:45}:" REDEFINED_LIFETIMEBOUND_MACRO"
  return a;
}

#undef REDEFINED_LIFETIMEBOUND_MACRO
#define REDEFINED_LIFETIMEBOUND_MACRO [[maybe_unused]]

View return_view_after_redefined_macro(View a) {
  // CHECK: :[[@LINE-1]]:46: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:46-[[@LINE-2]]:46}:" SECOND_LIFETIMEBOUND_MACRO"
  return a;
}

#define UNDEFINED_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

View return_view_with_undefined_macro(View a) {
  // CHECK: :[[@LINE-1]]:45: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:45-[[@LINE-2]]:45}:" UNDEFINED_LIFETIMEBOUND_MACRO"
  return a;
}

#undef UNDEFINED_LIFETIMEBOUND_MACRO

View return_view_after_undefined_macro(View a) {
  // CHECK: :[[@LINE-1]]:46: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:46-[[@LINE-2]]:46}:" SECOND_LIFETIMEBOUND_MACRO"
  return a;
}

struct MacroMember {
  MyObj data;

  View get_view() {
    // CHECK: :[[@LINE-1]]:18: warning: implicit this in intra-TU function should be marked
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:" SECOND_LIFETIMEBOUND_MACRO"
    // CHECK-MACRO: :[[@LINE-3]]:18: warning: implicit this in intra-TU function should be marked
    // CHECK-MACRO: fix-it:"{{.*}}":{[[@LINE-4]]:18-[[@LINE-4]]:18}:" LIFETIMEBOUND_MACRO"
    return data;
  }
};
