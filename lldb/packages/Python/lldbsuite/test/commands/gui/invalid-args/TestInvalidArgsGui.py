import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class GuiTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @no_debug_info_test
    def test_reproducer_generate_invalid_invocation(self):
        self.expect("gui blub", error=True,
                    substrs=["the gui command takes no arguments."])
