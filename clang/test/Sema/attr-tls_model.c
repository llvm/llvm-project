// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify -fsyntax-only %s

#if !__has_attribute(tls_model)
#error "Should support tls_model attribute"
#endif

int f(void) __attribute((tls_model("global-dynamic"))); // expected-error {{'tls_model' attribute only applies to thread-local variables}}

int x __attribute((tls_model("global-dynamic"))); // expected-error {{'tls_model' attribute only applies to thread-local variables}}
static __thread int y __attribute((tls_model("global-dynamic"))); // no-warning

static __thread int y __attribute((tls_model("local", "dynamic"))); // expected-error {{'tls_model' attribute takes one argument}}
static __thread int y __attribute((tls_model(123))); // expected-error {{expected string literal as argument of 'tls_model' attribute}}
static __thread int y __attribute((tls_model("foobar"))); // expected-error {{tls_model must be "global-dynamic", "local-dynamic", "initial-exec" or "local-exec"}}
