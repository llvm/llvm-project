// Tests that all-returns-nonnull inference works regardless of source order.
// The TU-level call-graph analysis processes callees before callers, so a
// function defined AFTER its caller still gets analyzed first.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify
// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nullable -std=c++17 %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/../Analysis/Scalable/NullabilitySafety/Inputs/decode-summary.py %t.json | FileCheck %s --check-prefix=EVIDENCE

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
Widget *make_widget() {
    return new Widget();
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

Widget *wrap_create() {
    return create();
}

Widget *create() {
    return new Widget();
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

    Widget *getWidget() {
        return &widget;
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

Widget *singleton() {
    static Widget instance;
    return &instance;
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
        return new Widget();
    return nullptr;
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
    if (!head) return nullptr;
    if (head->val % 2 == 0) return head;
    return find_odd(head->next);
}

ListNode *find_odd(ListNode *_Nullable head) {
    if (!head) return nullptr;
    if (head->val % 2 != 0) return head;
    return find_even(head->next);
}

// Callers of recursive functions should still warn
void use_recursive(ListNode *_Nullable head) {
    ListNode *n = find_even(head);
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

void takes_optional_ptr(Widget *w = nullptr) {
    if (w)
        w->x = 1;
}

void test_nullptr_default_evidence() {
    Widget w;
    takes_optional_ptr(&w);
}

// nullptr default with integer literal 0
void takes_ptr_zero_default(Widget *w = 0) {
    if (w)
        w->x = 2;
}

void test_zero_default_evidence() {
    Widget w;
    takes_ptr_zero_default(&w);
}

// Non-null default should NOT emit nullable evidence
Widget g_widget;
void takes_ptr_nonnull_default(Widget *w = &g_widget) {
    if (w)
        w->x = 3;
}

void test_nonnull_default_evidence() {
    Widget w;
    takes_ptr_nonnull_default(&w);
}

// EVIDENCE-NOT:  {{.}}
// EVIDENCE:      c:@F@create# AllReturnsNonnull c:@F@create# return
// EVIDENCE-NEXT: c:@F@create# NonnullEvidence c:@F@create# return
// EVIDENCE-NEXT: c:@F@find_even#*$@S@ListNode# ConditionalEvidence c:@F@find_even#*$@S@ListNode# return <- c:@F@find_odd#*$@S@ListNode# return
// EVIDENCE-NEXT: c:@F@find_even#*$@S@ListNode# MaybeNullEvidence c:@F@find_even#*$@S@ListNode# param 1
// EVIDENCE-NEXT: c:@F@find_even#*$@S@ListNode# NonnullEvidence c:@F@find_even#*$@S@ListNode# return
// EVIDENCE-NEXT: c:@F@find_even#*$@S@ListNode# NullableEvidence c:@F@find_even#*$@S@ListNode# return
// EVIDENCE-NEXT: c:@F@find_even#*$@S@ListNode# NullableEvidence c:@F@find_odd#*$@S@ListNode# param 1
// EVIDENCE-NEXT: c:@F@find_odd#*$@S@ListNode# ConditionalEvidence c:@F@find_odd#*$@S@ListNode# return <- c:@F@find_even#*$@S@ListNode# return
// EVIDENCE-NEXT: c:@F@find_odd#*$@S@ListNode# MaybeNullEvidence c:@F@find_odd#*$@S@ListNode# param 1
// EVIDENCE-NEXT: c:@F@find_odd#*$@S@ListNode# NonnullEvidence c:@F@find_odd#*$@S@ListNode# return
// EVIDENCE-NEXT: c:@F@find_odd#*$@S@ListNode# NullableEvidence c:@F@find_even#*$@S@ListNode# param 1
// EVIDENCE-NEXT: c:@F@find_odd#*$@S@ListNode# NullableEvidence c:@F@find_odd#*$@S@ListNode# return
// EVIDENCE-NEXT: c:@F@make_widget# AllReturnsNonnull c:@F@make_widget# return
// EVIDENCE-NEXT: c:@F@make_widget# NonnullEvidence c:@F@make_widget# return
// EVIDENCE-NEXT: c:@F@maybe_null#b# NonnullEvidence c:@F@maybe_null#b# return
// EVIDENCE-NEXT: c:@F@maybe_null#b# NullableEvidence c:@F@maybe_null#b# return
// EVIDENCE-NEXT: c:@F@singleton# AllReturnsNonnull c:@F@singleton# return
// EVIDENCE-NEXT: c:@F@singleton# NonnullEvidence c:@F@singleton# return
// EVIDENCE-NEXT: c:@F@takes_optional_ptr#*$@S@Widget# MaybeNullEvidence c:@F@takes_optional_ptr#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@takes_ptr_nonnull_default#*$@S@Widget# MaybeNullEvidence c:@F@takes_ptr_nonnull_default#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@takes_ptr_zero_default#*$@S@Widget# MaybeNullEvidence c:@F@takes_ptr_zero_default#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@test_nonnull_default_evidence# NonnullEvidence c:@F@takes_ptr_nonnull_default#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@test_nullptr_default_evidence# NonnullEvidence c:@F@takes_optional_ptr#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@test_nullptr_default_evidence# NullableEvidence c:@F@takes_optional_ptr#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@test_zero_default_evidence# NonnullEvidence c:@F@takes_ptr_zero_default#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@test_zero_default_evidence# NullableEvidence c:@F@takes_ptr_zero_default#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@use_recursive#*$@S@ListNode# NullableEvidence c:@F@find_even#*$@S@ListNode# param 1
// EVIDENCE-NEXT: c:@F@wrap_create# AllReturnsNonnull c:@F@wrap_create# return
// EVIDENCE-NEXT: c:@F@wrap_create# NonnullEvidence c:@F@wrap_create# return
// EVIDENCE-NEXT: c:@S@Factory@F@getWidget# AllReturnsNonnull c:@S@Factory@F@getWidget# return
// EVIDENCE-NEXT: c:@S@Factory@F@getWidget# NonnullEvidence c:@S@Factory@F@getWidget# return
// EVIDENCE-NEXT: c:@S@Holder@F@Holder#*$@S@Widget# ConditionalEvidence c:@S@Holder@FI@ptr <- c:@S@Holder@F@Holder#*$@S@Widget# param 1
// EVIDENCE-NOT:  {{.}}
