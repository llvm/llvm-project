// RUN: %clang_cc1 -verify -E --embed-dir=%S/Inputs -std=c2x %s

/* WG14 N3017: full
 * #embed - a scannable, tooling-friendly binary resource inclusion mechanism
 */

// C23 6.10p6
char b1[] = {
#embed "boop.h" limit(5)
,
#embed "boop.h" __limit__(5)
};
