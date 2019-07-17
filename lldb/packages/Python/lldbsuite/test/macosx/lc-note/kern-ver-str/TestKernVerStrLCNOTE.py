"""Test that corefiles with an LC_NOTE "kern ver str" load command is used."""

from __future__ import print_function


import os
import re
import subprocess
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestKernVerStrLCNOTE(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIfDarwinEmbedded
    @skipUnlessDarwin
    def test_lc_note(self):
        self.build()
        self.test_exe = self.getBuildArtifact("a.out")
        self.create_corefile = self.getBuildArtifact("create-empty-corefile")
        self.dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")
        self.corefile = self.getBuildArtifact("core")

        ## We can hook in our dsym-for-uuid shell script to lldb with this env
        ## var instead of requiring a defaults write.
        os.environ['LLDB_APPLE_DSYMFORUUID_EXECUTABLE'] = self.dsym_for_uuid
        self.addTearDownHook(lambda: os.environ.pop('LLDB_APPLE_DSYMFORUUID_EXECUTABLE', None))

        dwarfdump_uuid_regex = re.compile(
            'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')
        dwarfdump_cmd_output = subprocess.check_output(
                ('/usr/bin/dwarfdump --uuid "%s"' % self.test_exe), shell=True).decode("utf-8")
        aout_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                aout_uuid = match.group(1)
        self.assertNotEqual(aout_uuid, None, "Could not get uuid of built a.out")

        ###  Create our dsym-for-uuid shell script which returns self.test_exe
        ###  and its dSYM when given self.test_exe's UUID.
        shell_cmds = [
                '#! /bin/sh',
                'ret=0',
                'echo "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>"',
                'echo "<!DOCTYPE plist PUBLIC \\"-//Apple//DTD PLIST 1.0//EN\\" \\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\\">"',
                'echo "<plist version=\\"1.0\\">"',
                '',
                '# the last arugment is probably the uuid',
                'while [ $# -gt 1 ]',
                'do',
                '  shift',
                'done',
                'echo "<dict><key>$1</key><dict>"',
                '',
                'if [ "$1" = "%s" ]' % aout_uuid,
                'then',
                '  echo "<key>DBGArchitecture</key><string>x86_64</string>"',
                '  echo "<key>DBGDSYMPath</key><string>%s.dSYM/Contents/Resources/DWARF/%s</string>"' % (self.test_exe, os.path.basename(self.test_exe)),
                '  echo "<key>DBGSymbolRichExecutable</key><string>%s</string>"' % self.test_exe,
                'else',
                '  echo "<key>DBGError</key><string>not found</string>"',
                '  ret=1',
                'fi',
                'echo "</dict></dict></plist>"',
                'exit $ret'
                ]

        with open(self.dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + '\n')

        os.chmod(self.dsym_for_uuid, 0o755)

        ### Create our corefile
        retcode = call(self.create_corefile + " " +  self.corefile + " " + self.test_exe, shell=True)

        ### Now run lldb on the corefile
        ### which will give us a UUID
        ### which we call dsym-for-uuid.sh with
        ### which gives us a binary and dSYM
        ### which lldb should load!


        self.target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        self.process = self.target.LoadCore(self.corefile)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        filepath = fspec.GetDirectory() + "/" + fspec.GetFilename()
        self.assertEqual(filepath, self.test_exe)
