// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c++17 %s -verify
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -verify

void takes(int *_Nonnull);

void inner_nonnull(int *_Nullable p) {
  takes((int *)(int *_Nonnull)p);
}

void outer_nullable(int *_Nullable p) {
  takes((int *_Nullable)(int *_Nonnull)p); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

void outer_nullable_known_nonnull(int *_Nonnull p) {
  takes((int *_Nullable)(int *_Nonnull)p);
}

void outer_nonnull(int *_Nullable p) {
  takes((int *_Nonnull)(int *_Nullable)p);
}
