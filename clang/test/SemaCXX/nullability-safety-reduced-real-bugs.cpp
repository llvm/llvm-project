// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify

// Reduced from real null dereferences. Each must keep warning through every
// false-positive fix.

struct Name {};
struct Widget { void setName(const Name &); };
struct Loader {
  bool load(Widget &);
  bool loadInto(Widget *&);
  bool loadIntoPtr(Widget **);
};
void logError(const char *);

// The failure path nulls the pointer, then falls through to the use.
Widget *null_on_failure_then_use(Loader &loader, const Name &name) {
  Widget *w = new Widget();
  if (!loader.load(*w)) {
    delete w;
    w = nullptr;
    logError("load failed");
  }
  w->setName(name); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return w;
}

// An escape as an output parameter before the null store does not excuse the
// dereference after it.
Widget *escape_by_ref_before_null(Loader &loader, const Name &name) {
  Widget *w = new Widget();
  loader.loadInto(w);
  if (!loader.load(*w)) {
    delete w;
    w = nullptr;
  }
  w->setName(name); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return w;
}

Widget *escape_by_addr_before_null(Loader &loader, const Name &name) {
  Widget *w = new Widget();
  loader.loadIntoPtr(&w);
  if (!loader.load(*w)) {
    delete w;
    w = nullptr;
  }
  w->setName(name); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return w;
}

// The only assignment is compiled out, so the pointer stays null.
#define FEATURE_ENABLED 0
struct Part { void setLevel(int); };
struct Node { template <class T> T *create(); };

void configured_out_init(Node *node) {
  Part *part = nullptr;
#if FEATURE_ENABLED
  part = node->create<Part>();
#endif
  part->setLevel(0); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
