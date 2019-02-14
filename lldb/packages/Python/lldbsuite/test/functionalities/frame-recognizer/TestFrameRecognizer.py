# encoding: utf-8
"""
Test lldb's frame recognizers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import recognizer

class FrameRecognizerTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_frame_recognizer_1(self):
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.runCmd("command script import " + os.path.join(self.getSourceDir(), "recognizer.py"))

        self.expect("frame recognizer list",
                    substrs=['no matching results found.'])

        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo")

        self.expect("frame recognizer list",
                    substrs=['0: recognizer.MyFrameRecognizer, module a.out, function foo'])

        self.runCmd("frame recognizer add -l recognizer.MyOtherFrameRecognizer -s a.out -n bar -x")

        self.expect("frame recognizer list",
                    substrs=['0: recognizer.MyFrameRecognizer, module a.out, function foo',
                             '1: recognizer.MyOtherFrameRecognizer, module a.out, function bar (regexp)'
                    ])

        self.runCmd("frame recognizer delete 0")

        self.expect("frame recognizer list",
                    substrs=['1: recognizer.MyOtherFrameRecognizer, module a.out, function bar (regexp)'])

        self.runCmd("frame recognizer clear")

        self.expect("frame recognizer list",
                    substrs=['no matching results found.'])

        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo")

        lldbutil.run_break_set_by_symbol(self, "foo")
        self.runCmd("r")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        process = target.GetProcess()
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        self.assertEqual(frame.GetSymbol().GetName(), "foo")
        self.assertFalse(frame.GetLineEntry().IsValid())

        self.expect("frame variable",
                    substrs=['(int) a = 42', '(int) b = 56'])

        # Recognized arguments don't show up by default...
        variables = frame.GetVariables(lldb.SBVariablesOptions())
        self.assertEqual(variables.GetSize(), 0)

        # ...unless you set target.display-recognized-arguments to 1...
        self.runCmd("settings set target.display-recognized-arguments 1")
        variables = frame.GetVariables(lldb.SBVariablesOptions())
        self.assertEqual(variables.GetSize(), 2)

        # ...and you can reset it back to 0 to hide them again...
        self.runCmd("settings set target.display-recognized-arguments 0")
        variables = frame.GetVariables(lldb.SBVariablesOptions())
        self.assertEqual(variables.GetSize(), 0)

        # ... or explicitly ask for them with SetIncludeRecognizedArguments(True).
        opts = lldb.SBVariablesOptions()
        opts.SetIncludeRecognizedArguments(True)
        variables = frame.GetVariables(opts)

        self.assertEqual(variables.GetSize(), 2)
        self.assertEqual(variables.GetValueAtIndex(0).name, "a")
        self.assertEqual(variables.GetValueAtIndex(0).signed, 42)
        self.assertEqual(variables.GetValueAtIndex(0).GetValueType(), lldb.eValueTypeVariableArgument)
        self.assertEqual(variables.GetValueAtIndex(1).name, "b")
        self.assertEqual(variables.GetValueAtIndex(1).signed, 56)
        self.assertEqual(variables.GetValueAtIndex(1).GetValueType(), lldb.eValueTypeVariableArgument)

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by recognizer.MyFrameRecognizer'])

        self.expect("frame recognizer info 999", error=True,
                    substrs=['no frame with index 999'])

        self.expect("frame recognizer info 1",
                    substrs=['frame 1 not recognized by any recognizer'])

        # FIXME: The following doesn't work yet, but should be fixed.
        """
        lldbutil.run_break_set_by_symbol(self, "bar")
        self.runCmd("c")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect("frame variable -t",
                    substrs=['(int *) a = '])

        self.expect("frame variable -t *a",
                    substrs=['*a = 78'])
        """
