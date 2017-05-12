from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.expectedFailureAll(bugnumber="rdar://problem/32096064")])
