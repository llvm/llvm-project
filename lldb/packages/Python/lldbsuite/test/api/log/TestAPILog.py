"""
Test API logging.
"""

import re

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class APILogTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_api_log(self):
        """Test API logging"""
        logfile = os.path.join(self.getBuildDir(), "api-log.txt")

        def cleanup():
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)
        self.expect("log enable lldb api -f {}".format(logfile))

        self.dbg.SetDefaultArchitecture(None)
        self.dbg.GetScriptingLanguage(None)
        target = self.dbg.CreateTarget(None)

        print(logfile)
        with open(logfile, 'r') as f:
            log = f.read()

        # Find the debugger addr.
        debugger_addr = re.findall(
            r"lldb::SBDebugger::GetScriptingLanguage\(const char \*\) \(0x([0-9a-fA-F]+),",
            log)[0]

        get_scripting_language = 'lldb::ScriptLanguage lldb::SBDebugger::GetScriptingLanguage(const char *) (0x{}, "")'.format(
            debugger_addr)
        create_target = 'lldb::SBTarget lldb::SBDebugger::CreateTarget(const char *) (0x{}, "")'.format(
            debugger_addr)

        self.assertTrue(get_scripting_language in log, log)
        self.assertTrue(create_target in log, log)
