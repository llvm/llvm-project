import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestGDBRemoteClient(GDBRemoteTestBase):

    def test_connect(self):
        """Test connecting to a remote gdb server"""
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.assertPacketLogContains(["qProcessInfo", "qfThreadInfo"])
