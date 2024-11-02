// RUN: %clang_cc1 -verify -std=c11 %s
// expected-no-diagnostics

/* WG14 N1514: Yes
 * Conditional normative status for Annex G
 */

// We don't support Annex G (which introduces imaginary types), but support for
// this annex is conditional in C11. So we can test for conformance to this
// paper by ensuring we don't define the macro claiming we support Annex G.

#ifdef __STDC_IEC_559_COMPLEX__
#error "when did this happen??"
#endif
