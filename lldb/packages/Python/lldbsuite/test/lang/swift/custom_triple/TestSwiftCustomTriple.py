import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftCustomTriple(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=no_match(["linux"]))
    def test_custom_triple(self):
        """
        Test support for serialized custom triples. Unfortunately things
        like the vendor name can affect Swift's ability to find the
        correct stdlib.

        """
        self.build()
        logfile = self.getBuildArtifact('types.log')
        self.expect("log enable lldb types -f " + logfile)
        target, process, thread, bkpt = \
            lldbutil.run_to_name_breakpoint(self, 'main')
        self.expect("p 1")
        log = open(logfile, 'r')
        n = 0
        saw_aout_context = False
        saw_scratch_context = False
        expect_triple = False
        for line in log:
            if 'SwiftASTContext("a.out")::LogConfiguration' in line:
                expect_triple = True
                saw_aout_context = True
                continue
            if 'SwiftASTContextForExpressions::LogConfiguration' in line:
                expect_triple = True
                saw_scratch_context = True
                continue
            if expect_triple:
                self.assertTrue('Architecture' in line)
                self.assertTrue('x86_64-awesomevendor-linux' in line)
                expect_triple = False
                n += 1
        self.assertTrue(saw_aout_context)
        self.assertTrue(saw_scratch_context)
        self.assertEquals(n, 2)
                        
