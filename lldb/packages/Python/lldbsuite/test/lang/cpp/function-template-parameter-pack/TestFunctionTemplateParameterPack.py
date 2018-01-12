from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

# https://bugs.llvm.org/show_bug.cgi?id=35920
# This test stresses expression evaluation support for template functions.
# Currently the support is rudimentary, and running this test causes assertion
# failures in clang. This test cannot be XFAIL'ed because the test harness
# treats assertion failures as unexpected events. For now, the test must be
# skipped.
lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.skipIf])
