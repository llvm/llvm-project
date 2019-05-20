from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *

class TestNoWatchpointSupportInfo(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test(self):
        """
        Test lldb's parsing of the <architecture> tag in the target.xml register
        description packet.
        """
        class MyResponder(MockGDBServerResponder):

            def haltReason(self):
                return "T02thread:1ff0d;thread-pcs:10001bc00;"

            def threadStopInfo(self, threadnum):
                if threadnum == 0x1ff0d:
                    return "T02thread:1ff0d;thread-pcs:10001bc00;"

            def setBreakpoint(self, packet):
                if packet.startswith("Z2,"):
                    return "OK"

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
        if self.TraceOn():
            interp = self.dbg.GetCommandInterpreter()
            result = lldb.SBCommandReturnObject()
            interp.HandleCommand("log enable gdb-remote packets", result)
        self.dbg.SetDefaultArchitecture("x86_64")
        target = self.dbg.CreateTargetWithFileAndArch(None, None)

        process = self.connect(target)

        if self.TraceOn():
            interp = self.dbg.GetCommandInterpreter()
            result = lldb.SBCommandReturnObject()
            interp.HandleCommand("target list", result)
            print(result.GetOutput())

	
        err = lldb.SBError()
        wp = target.WatchAddress(0x100, 8, False, True, err)
        if self.TraceOn() and (err.Fail() or wp.IsValid == False):
            strm = lldb.SBStream()
            err.GetDescription(strm)
            print("watchpoint failed: %s" % strm.GetData())
        self.assertTrue(wp.IsValid())
