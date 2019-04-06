# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterKVO(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_kvo_with_run_command(self):
        """Test the behavior of formatters when KVO is in use."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set break point at this line.',
            lldb.SBFileSpec('main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # as long as KVO is implemented by subclassing, this test should succeed
        # we should be able to dynamically figure out that the KVO implementor class
        # is a subclass of Molecule, and use the appropriate summary for it
        self.runCmd("type summary add -s JustAMoleculeHere Molecule")
        self.expect('frame variable molecule', substrs=['JustAMoleculeHere'])
        self.runCmd("next")
        self.expect("thread list", substrs=['stopped', 'step over'])
        self.expect('frame variable molecule', substrs=['JustAMoleculeHere'])

        self.runCmd("next")
        # check that NSMutableDictionary's formatter is not confused when
        # dealing with a KVO'd dictionary
        self.expect(
            'frame variable newMutableDictionary',
            substrs=[
                '(NSDictionary *) newMutableDictionary = ',
                ' 21 key/value pairs'
            ])

        lldbutil.run_break_set_by_regexp(self, 'setAtoms')

        self.runCmd("continue")
        self.expect("frame variable _cmd", substrs=['setAtoms:'])
