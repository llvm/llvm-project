import lldb
import binascii
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestPlatformClient(GDBRemoteTestBase):

    def test_process_list(self):
        """Test connecting to a remote linux platform"""

        class MyResponder(MockGDBServerResponder):
            def qfProcessInfo(self, packet):
                if "all_users:1" in packet:
                    return "pid:10;ppid:1;uid:2;gid:3;euid:4;egid:5;name:" + binascii.hexlify("/a/process") + ";args:"
                else:
                    return "E04"

        self.server.responder = MyResponder()

        self.runCmd("platform select remote-linux")

        try:
            self.runCmd("platform connect connect://localhost:%d" %
                        self.server.port)
            self.assertTrue(self.dbg.GetSelectedPlatform().IsConnected())
            self.expect("platform process list -x",
                        startstr="1 matching process was found", endstr="process" + os.linesep)
            self.expect("platform process list -xv",
                        substrs=[
                            "PID    PARENT USER       GROUP      EFF USER   EFF GROUP",
                            "10     1      2          3          4          5"])
            self.expect("platform process list",
                        error="error: no processes were found on the \"remote-linux\" platform")
        finally:
            self.runCmd("platform disconnect")
