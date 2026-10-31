// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_dump(unsigned);
void clang_analyzer_eval(bool);

struct Msg {
  virtual unsigned cmd() const = 0;
};

namespace nonfinal_bifurcates {
// When the method is non-final and the dynamic type is unclear, the analysis
// should bifurcate, with one branch inlining the method and the other branch
// doing a conservative evaluation (which represents that another overriding
// method is called). (This is the baseline which is disabled in some cases.)
struct Ctrl : Msg {
  unsigned c;
  unsigned cmd() const override { return c; }
};

void test(Ctrl* p) {
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Element{SymRegion{reg_${{[0-9]+}}<Ctrl * p>},0 S64b,struct {{[0-9A-Za-z_]+}}::Ctrl}.c>}}
  // expected-warning@-2 {{conj_$}}
  clang_analyzer_eval(p->cmd() == p->cmd());
  // expected-warning@-1 {{TRUE}}
  // expected-warning@-2 {{FALSE}}
}
} // namespace nonfinal_bifurcates

namespace gh222960 {
// Ctrl::cmd() is final, the analyzer should not split off a "maybe dynamic
// dispatch invokes a different overriding method" execution path, and only
// follow the path where the method body is inlined.
struct Ctrl : Msg {
  unsigned c;
  unsigned cmd() const final { return c; }
};

void test(Ctrl* p) {
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Element{SymRegion{reg_${{[0-9]+}}<Ctrl * p>},0 S64b,struct {{[0-9A-Za-z_]+}}::Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd()); // expected-warning {{TRUE}}
}
} // namespace gh222960

namespace final_struct {
// The analyzer should also confidently inline the method of a final class.
struct Ctrl final : Msg {
  unsigned c;
  unsigned cmd() const override { return c; }
};

void test(Ctrl* p)
{
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Element{SymRegion{reg_${{[0-9]+}}<Ctrl * p>},0 S64b,struct {{[0-9A-Za-z_]+}}::Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd()); // expected-warning {{TRUE}}
}
} // namespace final_struct

namespace final_method_on_child_ptr {
// A final method should also be inlined when it is called through a pointer
// whose (static) type is a child of the class where it was defined.
struct Ctrl : Msg {
  unsigned c;
  unsigned cmd() const final { return c; }
};

struct Child : Ctrl {};

void test(Child* p) {
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Base{SymRegion{reg_${{[0-9]+}}<Child * p>},Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd()); // expected-warning {{TRUE}}
}
} // namespace final_method_on_child_ptr

namespace nonfinal_method_on_final_child_ptr {
// We can confidently inline even a non-final method of a non-final class if it
// is called on an object whose type is final and does not override it.
struct Ctrl : Msg {
  unsigned c;
  unsigned cmd() const override { return c; }
};

struct Child final : Ctrl {};

void test(Child* p) {
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Base{SymRegion{reg_${{[0-9]+}}<Child * p>},Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd()); // expected-warning {{TRUE}}
}
} // namespace nonfinal_method_on_final_child_ptr


namespace final_method_on_ptr_with_dyn_type_child {
// A final method should also be inlined when it is called through a pointer
// whose dynamic type is a child of the class where it was defined.
struct Ctrl : Msg {
  unsigned c;
  unsigned cmd() const final { return c; }
};

struct Child : Ctrl {};

void test(Child* childp) {
  Ctrl *p = childp;
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Base{SymRegion{reg_${{[0-9]+}}<Child * childp>},Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd());
  // expected-warning@-1 {{TRUE}}
}
} // namespace final_method_on_ptr_with_dyn_type_child

namespace final_method_on_base_ptr_with_known_dyn_type {
// A final method should also be inlined when it is called through a pointer
// base pointer with a known dynamic type.
struct Base : Msg {};

struct Ctrl : Base {
  unsigned c;
  unsigned cmd() const final { return c; }
};

void test(Ctrl* ctrlp) {
  Base *p = ctrlp;
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Element{SymRegion{reg_${{[0-9]+}}<Ctrl * ctrlp>},0 S64b,struct {{[0-9A-Za-z_]+}}::Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd());
  // expected-warning@-1 {{TRUE}}
}
} // namespace final_method_on_base_ptr_with_known_dyn_type

namespace nonfinal_method_on_ptr_with_dyn_type_final {
// We can confidently inline even a non-final method of a non-final class if it
// is called on an object whose dynamic type is final and does not override it.
struct Ctrl : Msg {
  unsigned c;
  unsigned cmd() const override { return c; }
};

struct Child final : Ctrl {};

void test(Child* childp) {
  Ctrl *p = childp;
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Base{SymRegion{reg_${{[0-9]+}}<Child * childp>},Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd());
  // expected-warning@-1 {{TRUE}}
}
} // namespace nonfinal_method_on_ptr_with_dyn_type_final
