// A copy of a smart pointer that is known non-null is itself non-null, and it
// owns its own reference: a later reset() or std::move() of the source does not
// null the copy. Copies of a maybe-null source stay maybe-null.
//
// UNSUPPORTED: target={{.*-windows.*}}
// REQUIRES: system-darwin || system-linux
// RUN: %clangxx -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -Xclang -verify

#include <memory>
#include <utility>

struct Widget {
  virtual ~Widget();
  void draw();
};
struct Button : Widget {};

std::shared_ptr<Widget> maybe_widget();

void copy_init() {
  auto s = std::make_shared<Widget>();
  std::shared_ptr<Widget> t = s;
  t->draw();
}

void copy_auto() {
  auto s = std::make_shared<Widget>();
  auto t = s;
  t->draw();
}

void copy_direct_init() {
  auto s = std::make_shared<Widget>();
  std::shared_ptr<Widget> t(s);
  t->draw();
}

void copy_assign() {
  auto s = std::make_shared<Widget>();
  std::shared_ptr<Widget> t;
  t = s;
  t->draw();
}

void copy_from_nonnull_param(std::shared_ptr<Widget> _Nonnull s) {
  std::shared_ptr<Widget> t = s;
  t->draw();
}

void copy_from_checked(std::shared_ptr<Widget> s) {
  if (!s)
    return;
  auto t = s;
  t->draw();
}

void copy_converting() {
  auto b = std::make_shared<Button>();
  std::shared_ptr<Widget> w = b;
  w->draw();
}

void copy_then_reset_source() {
  auto s = std::make_shared<Widget>();
  std::shared_ptr<Widget> t = s;
  s.reset();
  t->draw();
  s->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void copy_then_move_source() {
  auto s = std::make_shared<Widget>();
  auto t = s;
  auto u = std::move(s);
  t->draw();
  u->draw();
}

void copy_then_reset_copy() {
  auto s = std::make_shared<Widget>();
  auto t = s;
  t.reset();
  s->draw();
  t->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void copy_from_unchecked() {
  auto s = maybe_widget();
  auto t = s;
  t->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void copy_from_moved_from() {
  auto s = std::make_shared<Widget>();
  auto u = std::move(s);
  auto t = s;
  t->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void copy_assign_overwrites_proof(std::shared_ptr<Widget> other) {
  auto t = std::make_shared<Widget>();
  t = other;
  t->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void reference_is_not_a_copy() {
  auto p = std::make_shared<Widget>();
  auto &alias = p;
  p.reset();
  alias->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void const_reference_is_not_a_copy() {
  auto p = std::make_shared<Widget>();
  const std::shared_ptr<Widget> &alias = p;
  p.reset();
  alias->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void self_assign_keeps_proof() {
  auto p = std::make_shared<Widget>();
  p = p;
  p->draw();
}

// A reference to a smart pointer is the source object under another name, so
// every fact is shared: narrowing, reset() and moves through either name reach
// both.

void reference_deref_without_reset() {
  auto p = std::make_shared<Widget>();
  auto &alias = p;
  alias->draw();
}

void reset_through_reference() {
  auto p = std::make_shared<Widget>();
  auto &alias = p;
  alias.reset();
  p->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void reset_through_reference_then_alias() {
  auto p = std::make_shared<Widget>();
  auto &alias = p;
  alias.reset();
  alias->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void reference_to_reference() {
  auto p = std::make_shared<Widget>();
  auto &a = p;
  auto &b = a;
  b->draw();
  p.reset();
  b->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void check_through_reference(std::shared_ptr<Widget> p) {
  auto &alias = p;
  if (!alias)
    return;
  p->draw();
}

void rvalue_reference_does_not_move() {
  auto p = std::make_shared<Widget>();
  auto &&r = std::move(p);
  p->draw();
  r->draw();
}

void reassign_through_reference(std::shared_ptr<Widget> p) {
  auto &alias = p;
  alias = std::make_shared<Widget>();
  p->draw();
}

// Control: an alias to a maybe-null source warns with no reset anywhere. It is
// what distinguishes "the fact propagated" from "the alias is untracked".
void reference_to_nullable_source() {
  auto nullable = maybe_widget();
  auto &alias = nullable;
  alias->draw(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
