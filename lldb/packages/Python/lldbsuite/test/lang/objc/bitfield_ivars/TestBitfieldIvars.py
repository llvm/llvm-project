from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [
        decorators.skipIfFreeBSD,
        decorators.skipIfLinux,
        decorators.skipIfWindows,
        decorators.expectedFailureAll(
            bugnumber="rdar://problem/17990991")])
