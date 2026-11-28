// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=system -Wsystem-headers %s

int x;

# 12 // #1
// expected-warning@#1 {{this style of line directive is a GNU extension}}
// system-warning@#1 {{this style of line directive is a GNU extension}}
# 18 "foo" // #2
// expected-warning@#2 {{this style of line directive is a GNU extension}}
// system-warning@#2 {{this style of line directive is a GNU extension}}
# 24 "foo" 2 // #3
// expected-error@#3 {{invalid line marker flag '2': cannot pop empty include stack}}
// system-error@#3 {{invalid line marker flag '2': cannot pop empty include stack}}
# 30 "foo" 1 3  // #4: enter
// expected-warning@#4 {{this style of line directive is a GNU extension}}
// system-warning@#4 {{this style of line directive is a GNU extension}}
# 36 "foo" 2 3  // #5: exit
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#5 {{this style of line directive is a GNU extension}}
# 42 "foo" 2 3 4 // #6
// expected-error@#6 {{invalid line marker flag '2': cannot pop empty include stack}}
// system-error@#6 {{invalid line marker flag '2': cannot pop empty include stack}}
# 48 "foo" 3 4 // #7
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#7 {{this style of line directive is a GNU extension}}


// Verify that linemarker diddling of the system header flag works.

# 62 "glomp.h" // #8: in a system header.
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#8 {{this style of line directive is a GNU extension}}

# 70 "glomp.h" 3 // #9: not in a system header.
// expected-warning@#9 {{this style of line directive is a GNU extension}}
// system-warning@#9 {{this style of line directive is a GNU extension}}

#line 78 "blonk.h"  // doesn't change system headerness.

# 82     // #10: doesn't change system headerness.
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#10 {{this style of line directive is a GNU extension}}

# 90 "blonk.h"  // #11: DOES change system headerness.
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#11 {{this style of line directive is a GNU extension}}
