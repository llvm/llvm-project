// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -Wno-nullability-safety -std=c++17 %s -verify

int suppressedBefore(int *_Nullable p) {
  return *p;
}

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wnullability-safety-dereference"
int reenabledInSource(int *_Nullable p) {
  return *p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}
#pragma clang diagnostic pop

int suppressedAfter(int *_Nullable p) {
  return *p;
}
