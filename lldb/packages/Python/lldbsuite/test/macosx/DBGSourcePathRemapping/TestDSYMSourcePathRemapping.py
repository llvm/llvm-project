import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestDSYMSourcePathRemapping(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def build(self):
        botdir = self.getBuildArtifact('buildbot')
        userdir = self.getBuildArtifact('user')
        inputs = self.getSourcePath('Inputs')
        lldbutil.mkdir_p(botdir)
        lldbutil.mkdir_p(userdir)
        import shutil
        for f in ['main.c', 'relative.c']:
            shutil.copyfile(os.path.join(inputs, f), os.path.join(botdir, f))
            shutil.copyfile(os.path.join(inputs, f), os.path.join(userdir, f))

        super(TestDSYMSourcePathRemapping, self).build()

        # Remove the build sources.
        self.assertTrue(os.path.isdir(botdir))
        shutil.rmtree(botdir)

        # Create a plist.
        import subprocess
        dsym = self.getBuildArtifact('a.out.dSYM')
        uuid = subprocess.check_output(["/usr/bin/dwarfdump", "--uuid", dsym]
                                      ).decode("utf-8").split(" ")[1]
        import re
        self.assertTrue(re.match(r'[0-9a-fA-F-]+', uuid))
        plist = os.path.join(dsym, 'Contents', 'Resources', uuid + '.plist')
        with open(plist, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n')
            f.write('<plist version="1.0">\n')
            f.write('<dict>\n')
            f.write('  <key>DBGSourcePathRemapping</key>\n')
            f.write('  <dict>\n')
            f.write('    <key>' + botdir + '</key>\n')
            f.write('    <string>' + userdir + '</string>\n')
            f.write('  </dict>\n')
            f.write('</dict>\n')
            f.write('</plist>\n')


    @skipIf(debug_info=no_match("dsym"))
    def test(self):
        self.build()

        target, process, _, _ = lldbutil.run_to_name_breakpoint(
            self, 'main')
        self.expect("source list -n main", substrs=["Hello Absolute"])
        bkpt = target.BreakpointCreateByName('relative')
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("source list -n relative", substrs=["Hello Relative"])
