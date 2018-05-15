import lldb
import binascii
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestGDBRemoteClient(GDBRemoteTestBase):

    def test_connect(self):
        """Test connecting to a remote gdb server"""
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.assertPacketLogContains(["qProcessInfo", "qfThreadInfo"])

    def test_attach_fail(self):
        error_msg = "mock-error-msg"

        class MyResponder(MockGDBServerResponder):
            # Pretend we don't have any process during the initial queries.
            def qC(self):
                return "E42"

            def qfThreadInfo(self):
                return "OK" # No threads.

            # Then, when we are asked to attach, error out.
            def vAttach(self, pid):
                return "E42;" + binascii.hexlify(error_msg.encode()).decode()

        self.server.responder = MyResponder()

        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process, [lldb.eStateConnected])

        error = lldb.SBError()
        target.AttachToProcessWithID(lldb.SBListener(), 47, error)
        self.assertEquals(error_msg, error.GetCString())
