// FIXME: This import has no effect, because the submodule isn't built yet, and
// we don't map an @import to a #include in this case.
@import import_self.c;
#include "import-self-d.h"

// FIXME: This should not work; names from 'a' should not be visible here.
MyTypeA import_self_test_a;

// FIXME: This should work but does not; names from 'c' are not actually visible here.
//MyTypeC import_self_test_c;

MyTypeD import_self_test_d;
