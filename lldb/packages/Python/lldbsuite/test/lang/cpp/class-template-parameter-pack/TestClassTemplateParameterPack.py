from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.expectedFailureAll(
            compiler="gcc"),
        decorators.expectedFailureAll(
            bugnumber="rdar://47370292")])
