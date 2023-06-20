// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr8 -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-cpu pwr8 -verify -fsyntax-only %s

#if !__has_attribute(tls_model)
#error "Should support tls_model attribute"
#endif

static __thread int y __attribute((tls_model("global-dynamic"))); // no-warning
static __thread int y __attribute((tls_model("local-dynamic"))); // expected-error {{TLS model 'local-dynamic' is not yet supported on AIX}}
static __thread int y __attribute((tls_model("initial-exec"))); // expected-error {{TLS model 'initial-exec' is not yet supported on AIX}}
static __thread int y __attribute((tls_model("local-exec"))); // no-warning
