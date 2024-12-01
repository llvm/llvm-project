// RUN: %clang_cc1 "-triple" "arm64-apple-macosx10.15" -fsyntax-only -verify %s

__attribute__((availability(macos,introduced=11)))
inline bool try_acquire() {
  return true;
}

template <class T>
__attribute__((availability(macos,introduced=11)))
bool try_acquire_for(T duration) { // expected-note{{'try_acquire_for<int>' has been marked as being introduced in macOS 11 here, but the deployment target is macOS 10.15}}
  return try_acquire();
}

int main() {
  try_acquire_for(1); // expected-warning{{'try_acquire_for<int>' is only available on macOS 11 or newer}}
  // expected-note@-1{{enclose 'try_acquire_for<int>' in a __builtin_available check to silence this warning}}
}