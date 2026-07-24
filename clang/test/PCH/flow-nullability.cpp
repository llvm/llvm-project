// Round-trip: the flow-sensitive nullability LangOpts
// (-fflow-sensitive-nullability, -fnullability-default=nullable) must survive
// serialization into a PCH. We build the PCH with the flags and include it with
// the SAME flags; the flow analysis must still fire on the consumer's code,
// proving the LangOpts were restored from the PCH consumer side.

// Test without PCH (sanity: the warning fires when the header is textually
// included with the flags).
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -include %S/flow-nullability.h -verify %s

// Build the PCH with the flags, then include it with the SAME flags.
// RUN: %clang_cc1 -x c++-header -emit-pch -fflow-sensitive-nullability -fnullability-default=nullable -o %t %S/flow-nullability.h
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -include-pch %t -verify %s

int useNode(Node *_Nullable p) {
  return p->value; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int useGetInt() {
  return *getInt(); // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
