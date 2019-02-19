# TestSwiftRewriteClangPaths.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import shutil

class TestSwiftRewriteClangPaths(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    @add_test_categories(["swiftpr"])
    @skipIf(debug_info=no_match(["dsym"]))
    def testWithRemap(self):
        self.dotest(True)

    @skipUnlessDarwin
    @swiftTest
    @add_test_categories(["swiftpr"])
    @skipIf(debug_info=no_match(["dsym"]))
    def testWithoutRemap(self):
        self.dotest(False)

    def find_plist(self):
        import glob
        plist = self.getBuildArtifact("libFoo.dylib.dSYM/Contents/Resources/*.plist")
        lst = glob.glob(plist)
        self.assertTrue(len(lst) == 1)
        return lst[0]
        
    def dotest(self, remap):
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        # To ensure the module is rebuilt remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")

        botdir = os.path.realpath(self.getBuildArtifact("buildbot"))
        userdir = os.path.realpath(self.getBuildArtifact("user"))
        self.assertFalse(os.path.isdir(botdir))
        self.assertTrue(os.path.isdir(userdir))
        plist = self.find_plist()
        self.assertTrue(os.path.isfile(plist))
        if remap:
            self.runCmd("settings set target.source-map %s %s %s %s" %
                        (botdir, userdir, '/nonexisting-rootdir', userdir))
        else:
            # Also delete the remapping plist from the .dSYM to verify
            # that this doesn't work by happy accident without it.
            os.remove(plist)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Foo.swift'))

        if remap:
            comment = "returns correct value"
            self.expect("p foo", comment, substrs=["x", "23"])
            self.expect("p bar", comment, substrs=["y", "42"])
            self.expect("fr var foo", comment, substrs=["x", "23"])
            self.expect("fr var bar", comment, substrs=["y", "42"])
            self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

        # Scan through the types log.
        errs = 0
        found_iquote = 0
        found_f = 0
        found_i1 = 0
        found_i2 = 0
        found_rel = 0
        found_abs = 0
        logfile = open(log, "r")
        for line in logfile:
            self.assertFalse("remapped -iquote" in line)
            if " remapped " in line:
                if line[:-1].endswith('/user'): found_abs += 1;
                continue
            if "error: missing required module 'CFoo'" in line:
                errs += 1
                continue
            if 'user/iquote-path' in line: found_iquote += 1; continue
            if 'user/I-single'    in line: found_i1 += 1;     continue
            if 'user/I-double'    in line: found_i2 += 1;     continue
            if './iquote-path'    in line: found_rel += 1;    continue
            if './I-'             in line: found_rel += 1;    continue
            if '/user/Frameworks' in line: found_f += 1;      continue

        if remap:
            self.assertEqual(errs, 0, "expected no module import error")
            # Module context + scratch context.
            self.assertEqual(found_iquote, 2)
            self.assertEqual(found_i1, 2)
            self.assertEqual(found_i2, 2)
            self.assertEqual(found_f, 4)
            self.assertEqual(found_rel, 0)
            self.assertEqual(found_abs, 1)
        else:
            self.assertTrue(errs > 0, "expected module import error")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
