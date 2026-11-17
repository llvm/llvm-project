// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fsyntax-only -verify %s
// expected-no-diagnostics

// The value of a conversion to an enumeration with a fixed bool underlying type
// is the value of the corresponding conversion to bool (C23 6.7.3.3p16), so
// only zero maps to BF.

enum Bool : bool { BF, BT };

_Static_assert((enum Bool)0 == BF, "");
_Static_assert((enum Bool)1 == BT, "");
_Static_assert((enum Bool)2 == BT, "");
_Static_assert((enum Bool)-1 == BT, "");
