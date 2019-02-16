from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.expectedFailureAll(
            compiler="gcc"),
        decorators.expectedFailureAll(
            oslist=['ios', 'watchos', 'tvos', 'bridgeos'],
            bugnumber="rdar://problem/48128064: class template declaration unexpectedly shadowed by VarDecl on MacOS")])
