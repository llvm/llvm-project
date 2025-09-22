// RUN: %clang_cc1 -triple aarch64-linux-gnu -verify -emit-llvm-only %s -DCHECK_IMPLICIT_DEFAULT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -verify -emit-llvm-only %s -DCHECK_EXPLICIT_DEFAULT

#if defined(CHECK_IMPLICIT_DEFAULT)

int implicit_default_ok(void) { return 0; }
__attribute__((target_clones("aes", "lse"))) int implicit_default_ok(void) { return 1; }

int implicit_default_bad(void) { return 0; }
// expected-error@+2 {{definition with same mangled name 'implicit_default_bad.default' as another definition}}
// expected-note@-2 {{previous definition is here}}
__attribute__((target_clones("aes", "lse", "default"))) int implicit_default_bad(void) { return 1; }

#elif defined(CHECK_EXPLICIT_DEFAULT)

__attribute__((target_version("default"))) int explicit_default_ok(void) { return 0; }
__attribute__((target_clones("aes", "lse"))) int explicit_default_ok(void) { return 1; }

__attribute__((target_version("default"))) int explicit_default_bad(void) { return 0; }
// expected-error@+2 {{definition with same mangled name 'explicit_default_bad.default' as another definition}}
// expected-note@-2 {{previous definition is here}}
__attribute__((target_clones("aes", "lse", "default"))) int explicit_default_bad(void) { return 1; }

#endif
