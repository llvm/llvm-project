// RUN: %clang_analyze_cc1 -verify %s -analyzer-output=text \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Stream


#include "Inputs/system-header-simulator.h"
char *logDump();
bool coin();

[[noreturn]] void halt();

void assert(bool b) {
  if (!b)
    halt();
}

//===----------------------------------------------------------------------===//
// Report for which we expect NoOwnershipChangeVisitor to add a new note.
//===----------------------------------------------------------------------===//

namespace stream_opened_in_fn_call {
// TODO: AST analysis of sink would reveal that it doesn't intent to free the
// allocated memory, but in this instance, its also the only function with
// the ability to do so, we should see a note here.
void sink(FILE *f) {
}

void f() {
  sink(fopen("input.txt", "w"));
  // expected-note@-1{{Stream opened here}}
} // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
// expected-note@-1{{Opened stream never closed. Potential resource leak}}
} // namespace stream_opened_in_fn_call

namespace stream_passed_to_fn_call {

void expectedClose(FILE *f) {
  if (char *log = logDump()) { // expected-note{{Assuming 'log' is null}}
                               // expected-note@-1{{Taking false branch}}
    printf("%s", log);
    fclose(f);
  }
} // expected-note{{Returning without closing stream object or storing it for later release}}

void f() {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  if (!f) // expected-note{{'f' is non-null}}
          // expected-note@-1{{Taking false branch}}
    return;
  if (coin()) { // expected-note{{Assuming the condition is true}}
                // expected-note@-1{{Taking true branch}}
    expectedClose(f); // expected-note{{Calling 'expectedClose'}}
                      // expected-note@-1{{Returning from 'expectedClose'}}

    return; // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
            // expected-note@-1{{Opened stream never closed. Potential resource leak}}
  }
  fclose(f);
}
} // namespace stream_passed_to_fn_call

namespace stream_shared_with_ptr_of_shorter_lifetime {

void sink(FILE *f) {
  FILE *Q = f;
  if (coin()) // expected-note {{Assuming the condition is false}}
              // expected-note@-1 {{Taking false branch}}
    fclose(f);
  (void)Q;
} // expected-note{{Returning without closing stream object or storing it for later release}}

void foo() {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  if (!f) // expected-note{{'f' is non-null}}
          // expected-note@-1{{Taking false branch}}
    return;
  sink(f); // expected-note {{Calling 'sink'}}
           // expected-note@-1 {{Returning from 'sink'}}
} // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
// expected-note@-1{{Opened stream never closed. Potential resource leak}}

} // namespace stream_shared_with_ptr_of_shorter_lifetime

//===----------------------------------------------------------------------===//
// Report for which we *do not* expect NoOwnershipChangeVisitor add a new note,
// nor do we want it to.
//===----------------------------------------------------------------------===//

namespace stream_not_passed_to_fn_call {

void expectedClose(FILE *f) {
  if (char *log = logDump()) {
    printf("%s", log);
    fclose(f);
  }
}

void f(FILE *p) {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  if (!f) // expected-note{{'f' is non-null}}
          // expected-note@-1{{Taking false branch}}
    return;
  expectedClose(p); // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
                    // expected-note@-1{{Opened stream never closed. Potential resource leak}}
}
} // namespace stream_not_passed_to_fn_call

namespace stream_shared_with_ptr_of_same_lifetime {

void expectedClose(FILE *f, FILE **p) {
  // NOTE: Not a job of NoOwnershipChangeVisitor, but maybe this could be
  // highlighted still?
  *p = f;
}

void f() {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  FILE *p = NULL;
  if (!f) // expected-note{{'f' is non-null}}
          // expected-note@-1{{Taking false branch}}
    return;
  expectedClose(f, &p);
} // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
  // expected-note@-1{{Opened stream never closed. Potential resource leak}}
} // namespace stream_shared_with_ptr_of_same_lifetime

namespace stream_passed_into_fn_that_doesnt_intend_to_free {
void expectedClose(FILE *f) {
}

void f() {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  if (!f) // expected-note{{'f' is non-null}}
          // expected-note@-1{{Taking false branch}}
    return;
  expectedClose(f);

} // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
  // expected-note@-1{{Opened stream never closed. Potential resource leak}}
} // namespace stream_passed_into_fn_that_doesnt_intend_to_free

namespace stream_passed_into_fn_that_doesnt_intend_to_free2 {
void bar();

void expectedClose(FILE *f) {
  // Correctly realize that calling bar() doesn't mean that this function would
  // like to deallocate anything.
  bar();
}

void f() {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  if (!f) // expected-note{{'f' is non-null}}
          // expected-note@-1{{Taking false branch}}
    return;
  expectedClose(f);

} // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
  // expected-note@-1{{Opened stream never closed. Potential resource leak}}
} // namespace stream_passed_into_fn_that_doesnt_intend_to_free2

namespace streamstate_from_closed_to_open {

// StreamState of the symbol changed from nothing to Allocated. We don't want to
// emit notes when the RefKind changes in the stack frame.
static FILE *fopenWrapper() {
  FILE *f = fopen("input.txt", "w"); // expected-note{{Stream opened here}}
  assert(f);
  return f;
}
void use_ret() {
  FILE *v;
  v = fopenWrapper(); // expected-note {{Calling 'fopenWrapper'}}
                      // expected-note@-1{{Returning from 'fopenWrapper'}}

} // expected-warning{{Opened stream never closed. Potential resource leak [unix.Stream]}}
  // expected-note@-1{{Opened stream never closed. Potential resource leak}}

} // namespace streamstate_from_closed_to_open
