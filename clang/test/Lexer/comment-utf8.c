// RUN: %clang_cc1 -fsyntax-only %s -Winvalid-utf8 -verify
// expected-no-diagnostics


//§ § § 😀 你好 ©

/*§ § § 😀 你好 ©*/

/*
§ § § 😀 你好 ©©©
*/

/* § § § 😀 你好 © */
/*
    a longer comment to exerce the vectorized code path
    ----------------------------------------------------
    αααααααααααααααααααααα      // here is some unicode
    ----------------------------------------------------
    ----------------------------------------------------
*/
