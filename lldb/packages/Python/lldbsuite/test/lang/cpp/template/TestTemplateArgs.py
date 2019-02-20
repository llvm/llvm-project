"""
Test that C++ template classes that have integer parameters work correctly.

We must reconstruct the types correctly so the template types are correct
and display correctly, and also make sure the expression parser works and
is able the find all needed functions when evaluating expressions
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TemplateArgsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def prepareProcess(self):
        self.build()

        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints inside and outside methods that take pointers to the
        # containing struct.
        line = line_number('main.cpp', '// Breakpoint 1')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", line, num_expected_locations=1, loc_exact=True)

        arguments = None
        environment = None

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            arguments, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(
            process.GetState() == lldb.eStateStopped,
            PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)

        # Get frame for current thread
        return thread.GetSelectedFrame()

    def test_integer_args(self):
        frame = self.prepareProcess()

        testpos = frame.FindVariable('testpos')
        self.assertTrue(
            testpos.IsValid(),
            'make sure we find a local variabble named "testpos"')
        self.assertTrue(testpos.GetType().GetName() == 'TestObj<1>')

        expr_result = frame.EvaluateExpression("testpos.getArg()")
        self.assertTrue(
            expr_result.IsValid(),
            'got a valid expression result from expression "testpos.getArg()"')
        self.assertTrue(expr_result.GetValue() == "1", "testpos.getArg() == 1")
        self.assertTrue(
            expr_result.GetType().GetName() == "int",
            'expr_result.GetType().GetName() == "int"')

        testneg = frame.FindVariable('testneg')
        self.assertTrue(
            testneg.IsValid(),
            'make sure we find a local variabble named "testneg"')
        self.assertTrue(testneg.GetType().GetName() == 'TestObj<-1>')

        expr_result = frame.EvaluateExpression("testneg.getArg()")
        self.assertTrue(
            expr_result.IsValid(),
            'got a valid expression result from expression "testneg.getArg()"')
        self.assertTrue(
            expr_result.GetValue() == "-1",
            "testneg.getArg() == -1")
        self.assertTrue(
            expr_result.GetType().GetName() == "int",
            'expr_result.GetType().GetName() == "int"')

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test_template_template_args(self):
        frame = self.prepareProcess()

        c1 = frame.FindVariable('c1')
        self.assertTrue(
            c1.IsValid(),
            'make sure we find a local variabble named "c1"')
        self.assertTrue(c1.GetType().GetName() == 'C<float, T1>')
        f1 = c1.GetChildMemberWithName("V").GetChildAtIndex(0).GetChildMemberWithName("f")
        self.assertTrue(f1.GetType().GetName() == 'float')
        self.assertTrue(f1.GetValue() == '1.5')

        c2 = frame.FindVariable('c2')
        self.assertTrue(
            c2.IsValid(),
            'make sure we find a local variabble named "c2"')
        self.assertTrue(c2.GetType().GetName() == 'C<double, T1, T2>')
        f2 = c2.GetChildMemberWithName("V").GetChildAtIndex(0).GetChildMemberWithName("f")
        self.assertTrue(f2.GetType().GetName() == 'double')
        self.assertTrue(f2.GetValue() == '1.5')
        f3 = c2.GetChildMemberWithName("V").GetChildAtIndex(1).GetChildMemberWithName("f")
        self.assertTrue(f3.GetType().GetName() == 'double')
        self.assertTrue(f3.GetValue() == '2.5')
        f4 = c2.GetChildMemberWithName("V").GetChildAtIndex(1).GetChildMemberWithName("i")
        self.assertTrue(f4.GetType().GetName() == 'int')
        self.assertTrue(f4.GetValue() == '42')

    # Gcc does not generate the necessary DWARF attribute for enum template
    # parameters.
    @expectedFailureAll(bugnumber="llvm.org/pr28354", compiler="gcc")
    @skipIf(dwarf_version=['<', '4'])
    def test_enum_args(self):
        frame = self.prepareProcess()

        # Make sure "member" can be displayed and also used in an expression
        # correctly
        member = frame.FindVariable('member')
        self.assertTrue(
            member.IsValid(),
            'make sure we find a local variabble named "member"')
        self.assertTrue(member.GetType().GetName() ==
                        'EnumTemplate<EnumType::Member>')

        expr_result = frame.EvaluateExpression("member.getMember()")
        self.assertTrue(
            expr_result.IsValid(),
            'got a valid expression result from expression "member.getMember()"')
        self.assertTrue(
            expr_result.GetValue() == "123",
            "member.getMember() == 123")
        self.assertTrue(
            expr_result.GetType().GetName() == "int",
            'expr_result.GetType().GetName() == "int"')

        # Make sure "subclass" can be displayed and also used in an expression
        # correctly
        subclass = frame.FindVariable('subclass')
        self.assertTrue(
            subclass.IsValid(),
            'make sure we find a local variabble named "subclass"')
        self.assertTrue(subclass.GetType().GetName() ==
                        'EnumTemplate<EnumType::Subclass>')

        expr_result = frame.EvaluateExpression("subclass.getMember()")
        self.assertTrue(
            expr_result.IsValid(),
            'got a valid expression result from expression "subclass.getMember()"')
        self.assertTrue(
            expr_result.GetValue() == "246",
            "subclass.getMember() == 246")
        self.assertTrue(
            expr_result.GetType().GetName() == "int",
            'expr_result.GetType().GetName() == "int"')
