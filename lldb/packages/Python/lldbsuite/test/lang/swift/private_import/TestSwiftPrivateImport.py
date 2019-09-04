import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftPrivateImport(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    def test_private_import(self):
        """Test a library with a private import for which there is no debug info"""
        invisible_swift = self.getBuildArtifact("Invisible.swift")
        import shutil
        shutil.copyfile("InvisibleSource.swift", invisible_swift)
        self.build()
        os.unlink(invisible_swift)
        os.unlink(self.getBuildArtifact("Invisible.swiftmodule"))
        os.unlink(self.getBuildArtifact("Invisible.swiftinterface"))
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        # We should not be able to resolve the types.
        self.expect("fr var -d run -- x", substrs=["(Any)"])
        # FIXME: This crashes LLDB with a Swift DESERIALIZATION FAILURE.
        # self.expect("fr var -d run -- y", substrs=["(Any)"])
