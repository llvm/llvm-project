from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestStopPCs(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return "T02thread:1ff0d;threads:1ff0d,2ff0d;thread-pcs:10001bc00,10002bc00;"

            def threadStopInfo(self, threadnum):
                if threadnum == 0x1ff0d:
                    return "T02thread:1ff0d;threads:1ff0d,2ff0d;thread-pcs:10001bc00,10002bc00;"
                if threadnum == 0x2ff0d:
                    return "T00thread:2ff0d;threads:1ff0d,2ff0d;thread-pcs:10001bc00,10002bc00;"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                          </feature>
                        </target>""", False
                else:
                    return None, False

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
          self.runCmd("log enable gdb-remote packets")
        process = self.connect(target)

        self.assertEqual(process.GetNumThreads(), 2)
        th0 = process.GetThreadAtIndex(0)
        th1 = process.GetThreadAtIndex(1)
        self.assertEqual(th0.GetThreadID(), 0x1ff0d)
        self.assertEqual(th1.GetThreadID(), 0x2ff0d)
        self.assertEqual(th0.GetFrameAtIndex(0).GetPC(), 0x10001bc00)
        self.assertEqual(th1.GetFrameAtIndex(0).GetPC(), 0x10002bc00)
