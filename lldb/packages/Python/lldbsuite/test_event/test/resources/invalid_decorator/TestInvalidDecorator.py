from __future__ import print_function
from lldbsuite.test import lldbtest
from lldbsuite.test import decorators


class NonExistentDecoratorTestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.nonExistentDecorator(bugnumber="yt/1300")
    def test(self):
        """Verify non-existent decorators are picked up by test runner."""
        pass
