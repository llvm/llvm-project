// Tests that all-returns-nonnull inference works regardless of source order.
// The TU-level call-graph analysis processes callees before callers, so a
// function defined AFTER its caller still gets analyzed first.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -Rnullsafe-evidence %s -verify

// ===----------------------------------------------------------------------===//
// Common types
// ===----------------------------------------------------------------------===//

struct Widget {
    int x;
};

// ===----------------------------------------------------------------------===//
// Order-independence: caller BEFORE callee
// ===----------------------------------------------------------------------===//

// Forward declaration only — definition comes later.
Widget *make_widget();

// Caller appears first in the file. Without call-graph ordering, the analysis
// wouldn't know make_widget always returns nonnull at this point.
void use_widget() {
    Widget *w = make_widget();
    w->x = 42; // OK - make_widget always returns nonnull (analyzed via call graph)
}

// Callee defined after its caller.
Widget *make_widget() { // expected-remark{{function 'make_widget' always returns a non-null pointer}}
    return new Widget(); // expected-remark-re{{function 'make_widget' of global scope (declared at {{.*}}) returns nonnull}}
}

// ===----------------------------------------------------------------------===//
// Transitive: A calls B calls C, all defined in reverse order
// ===----------------------------------------------------------------------===//

Widget *create();
Widget *wrap_create();

// A uses wrap_create (which uses create) — both defined below.
void top_level_user() {
    Widget *w = wrap_create();
    w->x = 1; // OK - transitive nonnull through call graph
}

Widget *wrap_create() { // expected-remark{{function 'wrap_create' always returns a non-null pointer}}
    return create(); // expected-remark-re{{function 'wrap_create' of global scope (declared at {{.*}}) returns nonnull}}
}

Widget *create() { // expected-remark{{function 'create' always returns a non-null pointer}}
    return new Widget(); // expected-remark-re{{function 'create' of global scope (declared at {{.*}}) returns nonnull}}
}

// ===----------------------------------------------------------------------===//
// Methods: caller uses method defined later in the class
// ===----------------------------------------------------------------------===//

struct Factory {
    Widget widget;

    // Caller method — uses getWidget defined below.
    void use() {
        Widget *w = getWidget();
        w->x = 10; // OK - getWidget always returns nonnull
    }

    Widget *getWidget() { // expected-remark{{function 'getWidget' always returns a non-null pointer}}
        return &widget; // expected-remark-re{{function 'getWidget' of Factory (declared at {{.*}}) returns nonnull}}
    }
};

// ===----------------------------------------------------------------------===//
// Mixed: some callers before, some after — all should work
// ===----------------------------------------------------------------------===//

Widget *singleton();

void early_caller() {
    Widget *w = singleton();
    w->x = 1; // OK
}

Widget *singleton() { // expected-remark{{function 'singleton' always returns a non-null pointer}}
    static Widget instance;
    return &instance; // expected-remark-re{{function 'singleton' of global scope (declared at {{.*}}) returns nonnull}}
}

void late_caller() {
    Widget *w = singleton();
    w->x = 2; // OK
}

// ===----------------------------------------------------------------------===//
// Negative: function that DOES return null should still warn callers
// ===----------------------------------------------------------------------===//

Widget *maybe_null(bool flag);

void caller_of_maybe_null() {
    Widget *w = maybe_null(true);
    w->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

Widget *maybe_null(bool flag) {
    if (flag)
        return new Widget(); // expected-remark{{returns nonnull}}
    return nullptr; // expected-remark{{returns nullable}}
}

// ===----------------------------------------------------------------------===//
// Recursive functions: can't infer all-returns-nonnull for recursive cycles,
// but should not crash or infinite loop
// ===----------------------------------------------------------------------===//

struct ListNode {
    int val;
    ListNode * _Nullable next;
};

// Mutually recursive — neither should get all-returns-nonnull
ListNode *find_even(ListNode *_Nullable head);
ListNode *find_odd(ListNode *_Nullable head);

ListNode *find_even(ListNode *_Nullable head) {
    if (!head) return nullptr; // expected-remark{{returns nullable}}
    if (head->val % 2 == 0) return head; // expected-remark{{returns nonnull}}
    return find_odd(head->next); // expected-remark-re{{parameter 'head' of 'find_odd' (declared at {{.*}}) called with nullable argument}}
}

ListNode *find_odd(ListNode *_Nullable head) {
    if (!head) return nullptr; // expected-remark{{returns nullable}}
    if (head->val % 2 != 0) return head; // expected-remark{{returns nonnull}}
    return find_even(head->next); // expected-remark-re{{parameter 'head' of 'find_even' (declared at {{.*}}) called with nullable argument}}
}

// Callers of recursive functions should still warn
void use_recursive(ListNode *_Nullable head) {
    ListNode *n = find_even(head); // expected-remark-re{{parameter 'head' of 'find_even' (declared at {{.*}}) called with nullable argument}}
    n->val = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// ===----------------------------------------------------------------------===//
// Constructor parameter evidence: CXXConstructorName is not an Identifier,
// so the evidence emission must accept non-identifier declaration names too.
// ===----------------------------------------------------------------------===//

struct Holder {
    Widget *ptr;
    Holder(Widget *p) : ptr(p) {} // unannotated param — no evidence
};

void test_constructor_param_evidence() {
    Widget w;
    Holder h(&w);
}

void test_constructor_param_evidence_nullable() {
    Widget *maybe = nullptr;
    Holder h(maybe);
}

// ===----------------------------------------------------------------------===//
// Parameter with nullptr default: the default value counts as nullable evidence
// even when every explicit caller passes nonnull.
// ===----------------------------------------------------------------------===//

void takes_optional_ptr(Widget *w = nullptr) { // expected-remark-re{{parameter 'w' of 'takes_optional_ptr' (declared at {{.*}}) called with nullable argument}}
    if (w)
        w->x = 1;
}

void test_nullptr_default_evidence() {
    Widget w;
    takes_optional_ptr(&w); // expected-remark-re{{parameter 'w' of 'takes_optional_ptr' (declared at {{.*}}) called with nonnull argument}}
}

// nullptr default with integer literal 0
void takes_ptr_zero_default(Widget *w = 0) { // expected-remark-re{{parameter 'w' of 'takes_ptr_zero_default' (declared at {{.*}}) called with nullable argument}}
    if (w)
        w->x = 2;
}

void test_zero_default_evidence() {
    Widget w;
    takes_ptr_zero_default(&w); // expected-remark-re{{parameter 'w' of 'takes_ptr_zero_default' (declared at {{.*}}) called with nonnull argument}}
}

// Non-null default should NOT emit nullable evidence
Widget g_widget;
void takes_ptr_nonnull_default(Widget *w = &g_widget) {
    if (w)
        w->x = 3;
}

void test_nonnull_default_evidence() {
    Widget w;
    takes_ptr_nonnull_default(&w); // expected-remark-re{{parameter 'w' of 'takes_ptr_nonnull_default' (declared at {{.*}}) called with nonnull argument}}
}
