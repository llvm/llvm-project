"""
Test that objective-c constant strings are generated correctly by the expression
parser.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ConstStringTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    d = {'OBJC_SOURCES': 'const-strings.m'}

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "const-strings.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    @skipUnlessDarwin
    def test_break(self):
        """Test constant string generation amd comparison by the expression parser."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(self.d)

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            self.main_source,
            self.line,
            num_expected_locations=1,
            loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=[" at %s:%d" % (self.main_source, self.line),
                             "stop reason = breakpoint"])

        self.expect('expression (int)[str compare:@"hello"]',
                    startstr="(int) $0 = 0")
        self.expect('expression (int)[str compare:@"world"]',
                    startstr="(int) $1 = -1")

        # Test empty strings, too.
        self.expect('expression (int)[@"" length]',
                    startstr="(int) $2 = 0")

        self.expect('expression (int)[@"123" length]',
                    startstr="(int) $3 = 3")
