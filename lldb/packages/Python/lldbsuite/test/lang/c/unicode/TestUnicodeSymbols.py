# coding=utf8
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestUnicodeSymbols(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_union_members(self):
        self.build()
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(self.getBuildArtifact("a.out")))
        module = lldb.SBModule(spec)
        self.assertTrue(module.IsValid())
        mytype = module.FindFirstType("foobár")
        self.assertTrue(mytype.IsValid())
        self.assertTrue(mytype.IsPointerType())
