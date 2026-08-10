// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=system -Wsystem-headers %s

int x;

# 42 // #1
// expected-warning@#1 {{this style of line directive is a GNU extension}}
// system-warning@#1 {{this style of line directive is a GNU extension}}
# 42 "foo" // #2
// expected-warning@#2 {{this style of line directive is a GNU extension}}
// system-warning@#2 {{this style of line directive is a GNU extension}}
# 42 "foo" 2 // #3
// expected-error@#3 {{invalid line marker flag '2': cannot pop empty include stack}}
// system-error@#3 {{invalid line marker flag '2': cannot pop empty include stack}}
# 42 "foo" 1 3  // #4: enter
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#4 {{this style of line directive is a GNU extension}}
# 42 "foo" 2 3  // #5: exit
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#5 {{this style of line directive is a GNU extension}}
# 42 "foo" 2 3 4 // #6
// expected-error@#6 {{invalid line marker flag '2': cannot pop empty include stack}}
// system-error@#6 {{invalid line marker flag '2': cannot pop empty include stack}}
# 42 "foo" 3 4 // #7
// expected-warning@#7 {{this style of line directive is a GNU extension}}
// system-warning@#7 {{this style of line directive is a GNU extension}}


// Verify that linemarker diddling of the system header flag works.

# 192 "glomp.h" // #8: not a system header.
// expected-warning@#8 {{this style of line directive is a GNU extension}}
// system-warning@#8 {{this style of line directive is a GNU extension}}

# 192 "glomp.h" 3 // #9: System header.
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#9 {{this style of line directive is a GNU extension}}

#line 42 "blonk.h"  // doesn't change system headerness.

# 97     // #10: doesn't change system headerness.
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#10 {{this style of line directive is a GNU extension}}

# 42 "blonk.h"  // #11: DOES change system headerness.
// Warnings silenced when -Wsystem-headers isn't passed.
// system-warning@#11 {{this style of line directive is a GNU extension}}
