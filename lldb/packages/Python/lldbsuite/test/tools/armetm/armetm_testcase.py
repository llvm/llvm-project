from lldbsuite.test.lldbtest import *
import os
import time
import json

ADDRESS_REGEX = "0x[0-9a-fA-F]*"


# Decorator that runs a test with both modes of USE_SB_API.
# It assumes that no tests can be executed in parallel.
def testSBAPIAndCommands(func):
    def wrapper(*args, **kwargs):
        TraceArmETMTestCaseBase.USE_SB_API = True
        func(*args, **kwargs)
        TraceArmETMTestCaseBase.USE_SB_API = False
        func(*args, **kwargs)

    return wrapper


# Class that should be used by all python Arm ETM tests.
#
# It has a handy check that skips the test if the arm-etm plugin is not enabled.
#
# It also contains many functions that can test both the SB API or the command line version
# of the most important tracing actions.
class TraceArmETMTestCaseBase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # If True, the trace test methods will use the SB API, otherwise they'll use raw commands.
    USE_SB_API = False

    def setUp(self):
        TestBase.setUp(self)
        if "arm-etm" not in configuration.enabled_plugins:
            self.skipTest("The arm-etm test plugin is not enabled")

    def getTraceOrCreate(self):
        if not self.target().GetTrace().IsValid():
            error = lldb.SBError()
            self.target().CreateTrace(error)
        return self.target().GetTrace()

    def assertSBError(self, sberror, error=False):
        if error:
            self.assertTrue(sberror.Fail())
        else:
            self.assertSuccess(sberror)

    def traceLoad(self, traceDescriptionFilePath, error=False, substrs=None):
        if self.USE_SB_API:
            traceDescriptionFile = lldb.SBFileSpec(traceDescriptionFilePath, True)
            loadTraceError = lldb.SBError()
            self.dbg.LoadTraceFromFile(loadTraceError, traceDescriptionFile)
            self.assertSBError(loadTraceError, error)
        else:
            command = f"trace load -v {traceDescriptionFilePath}"
            self.expect(command, error=error, substrs=substrs)
