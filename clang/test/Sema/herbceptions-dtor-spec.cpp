// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fsyntax-only -verify %s

// Destructors cannot be declared with a herbceptions exception specification:
// destruction must be able to run during cleanup, so it cannot itself fail
// through the throws/fails channel.

struct DtorThrows {
  ~DtorThrows() throws; // expected-error {{destructor cannot be declared with a herbceptions ('throws'/'return_failure{...}') exception specification; destructors cannot propagate errors}}
};

struct DtorThrowsTrue {
  ~DtorThrowsTrue() throws(true); // expected-error {{destructor cannot be declared with a herbceptions ('throws'/'return_failure{...}') exception specification; destructors cannot propagate errors}}
};

struct DtorFails {
  ~DtorFails() return_failure{int}; // expected-error {{destructor cannot be declared with a herbceptions ('throws'/'return_failure{...}') exception specification; destructors cannot propagate errors}} \
                          // expected-error {{'return_failure{...}' is a C-style feature and may only be attached to free (non-member) functions; it is not allowed on member functions (including static members), lambdas, or function templates}}
};

// noexcept destructors remain fine.
struct DtorOk {
  ~DtorOk() noexcept;
};
