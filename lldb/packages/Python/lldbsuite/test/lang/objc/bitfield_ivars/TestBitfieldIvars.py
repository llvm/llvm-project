from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [
        # This is a Darwin-only failure related to incorrect expresssion-
        # evaluation for single-bit ObjC bitfields.
        decorators.skipUnlessDarwin,
        decorators.expectedFailureAll(
            bugnumber="rdar://problem/17990991")])
