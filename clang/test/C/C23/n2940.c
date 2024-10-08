// RUN: %clang_cc1 -verify=no-trigraphs -std=c23 %s
// RUN: %clang_cc1 -verify=no-trigraphs -std=gnu23 %s
// RUN: %clang_cc1 -verify=no-trigraphs -std=gnu17 %s
// RUN: %clang_cc1 -verify=no-trigraphs -std=gnu11 %s
// RUN: %clang_cc1 -verify=no-trigraphs -std=gnu99 %s
// RUN: %clang_cc1 -verify=no-trigraphs -std=gnu89 %s
// RUN: %clang_cc1 -verify=trigraphs -std=c17 %s
// RUN: %clang_cc1 -verify=trigraphs -std=c11 %s
// RUN: %clang_cc1 -verify=trigraphs -std=c99 %s
// RUN: %clang_cc1 -verify=trigraphs -std=c89 %s
// RUN: %clang_cc1 -verify=trigraphs -std=c23 -ftrigraphs %s

/* WG14 N2940: Clang 18
 * Removing trigraphs??!
 */

// Trigraphs are enabled by default in any conforming C mode before C23, but
// are otherwise disabled (in all GNU modes, and in C23 or later).
// The ??= trigraph, if supported, will become the # character, which is a null
// preprocessor directive that does nothing.

??=
// no-trigraphs-warning@-1 {{trigraph ignored}} \
   no-trigraphs-error@-1 {{expected identifier or '('}} \
   trigraphs-warning@-1 {{trigraph converted to '#' character}}

