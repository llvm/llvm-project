"""
Test lldb-mi -stack-xxx commands.
"""

from __future__ import print_function



import lldbmi_testcase
from lldbsuite.test.lldbtest import *

class MiStackTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_arguments(self):
        """Test that 'lldb-mi --interpreter' can shows arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -stack-list-arguments lists empty stack arguments if range is empty
        self.runCmd("-stack-list-arguments 0 1 0")
        self.expect("\^done,stack-args=\[\]")

        # Test that -stack-list-arguments lists stack arguments without values
        # (and that low-frame and high-frame are optional)
        self.runCmd("-stack-list-arguments 0")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[name=\"argc\",name=\"argv\"\]}")
        self.runCmd("-stack-list-arguments --no-values")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[name=\"argc\",name=\"argv\"\]}")

        # Test that -stack-list-arguments lists stack arguments with all values
        self.runCmd("-stack-list-arguments 1 0 0")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}\]")
        self.runCmd("-stack-list-arguments --all-values 0 0")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",value=\"1\"},{name=\"argv\",value=\".*\"}\]}\]")

        # Test that -stack-list-arguments lists stack arguments with simple values
        self.runCmd("-stack-list-arguments 2 0 1")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",type=\"int\",value=\"1\"},{name=\"argv\",type=\"const char \*\*\",value=\".*\"}\]}")
        self.runCmd("-stack-list-arguments --simple-values 0 1")
        self.expect("\^done,stack-args=\[frame={level=\"0\",args=\[{name=\"argc\",type=\"int\",value=\"1\"},{name=\"argv\",type=\"const char \*\*\",value=\".*\"}\]}")

        # Test that an invalid low-frame is handled 
        # FIXME: -1 is treated as unsigned int
        self.runCmd("-stack-list-arguments 0 -1 0")
        #self.expect("\^error")
        self.runCmd("-stack-list-arguments 0 0")
        self.expect("\^error,msg=\"Command 'stack-list-arguments'\. Thread frame range invalid\"")

        # Test that an invalid high-frame is handled
        # FIXME: -1 is treated as unsigned int
        self.runCmd("-stack-list-arguments 0 0 -1")
        #self.expect("\^error")

        # Test that a missing low-frame or high-frame is handled
        self.runCmd("-stack-list-arguments 0 0")
        self.expect("\^error,msg=\"Command 'stack-list-arguments'\. Thread frame range invalid\"")

        # Test that an invalid low-frame is handled 
        self.runCmd("-stack-list-arguments 0 0")
        self.expect("\^error,msg=\"Command 'stack-list-arguments'\. Thread frame range invalid\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_locals(self):
        """Test that 'lldb-mi --interpreter' can shows local variables."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test int local variables:
        # Run to BP_local_int_test
        line = line_number('main.cpp', '// BP_local_int_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"a\",name=\"b\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"a\",name=\"b\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[{name=\"a\",type=\"int\",value=\"10\"},{name=\"b\",type=\"int\",value=\"20\"}\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[{name=\"a\",type=\"int\",value=\"10\"},{name=\"b\",type=\"int\",value=\"20\"}\]")
        
        # Test struct local variable:
        # Run to BP_local_struct_test
        line = line_number('main.cpp', '// BP_local_struct_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"3\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"var_c\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"var_c\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"var_c\",value=\"{var_a = 10, var_b = 97 'a', inner_ = {var_d = 30}}\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"var_c\",value=\"{var_a = 10, var_b = 97 'a', inner_ = {var_d = 30}}\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[{name=\"var_c\",type=\"my_type\"}\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[{name=\"var_c\",type=\"my_type\"}\]")
        
        # Test array local variable:
        # Run to BP_local_array_test
        line = line_number('main.cpp', '// BP_local_array_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"4\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"array\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"array\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"array\",value=\"{\[0\] = 100, \[1\] = 200, \[2\] = 300}\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"array\",value=\"{\[0\] = 100, \[1\] = 200, \[2\] = 300}\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[{name=\"array\",type=\"int \[3\]\"}\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[{name=\"array\",type=\"int \[3\]\"}\]")
        
        # Test pointers as local variable:
        # Run to BP_local_pointer_test
        line = line_number('main.cpp', '// BP_local_pointer_test')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"5\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-locals: use 0 or --no-values
        self.runCmd("-stack-list-locals 0")
        self.expect("\^done,locals=\[name=\"test_str\",name=\"var_e\",name=\"ptr\"\]")
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"test_str\",name=\"var_e\",name=\"ptr\"\]")

        # Test -stack-list-locals: use 1 or --all-values
        self.runCmd("-stack-list-locals 1")
        self.expect("\^done,locals=\[{name=\"test_str\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*?\"}\]")
        self.runCmd("-stack-list-locals --all-values")
        self.expect("\^done,locals=\[{name=\"test_str\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*?\"}\]")

        # Test -stack-list-locals: use 2 or --simple-values
        self.runCmd("-stack-list-locals 2")
        self.expect("\^done,locals=\[{name=\"test_str\",type=\"const char \*\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",type=\"int\",value=\"24\"},{name=\"ptr\",type=\"int \*\",value=\".*?\"}\]")
        self.runCmd("-stack-list-locals --simple-values")
        self.expect("\^done,locals=\[{name=\"test_str\",type=\"const char \*\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",type=\"int\",value=\"24\"},{name=\"ptr\",type=\"int \*\",value=\".*?\"}\]")
        
        # Test -stack-list-locals in a function with catch clause, 
        # having unnamed parameter
        # Run to BP_catch_unnamed
        line = line_number('main.cpp', '// BP_catch_unnamed')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"6\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -stack-list-locals: use --no-values
        self.runCmd("-stack-list-locals --no-values")
        self.expect("\^done,locals=\[name=\"i\",name=\"j\"\]")
    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_variables(self):
        """Test that 'lldb-mi --interpreter' can shows local variables and arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test int local variables:
        # Run to BP_local_int_test
        line = line_number('main.cpp', '// BP_local_int_test_with_args')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -stack-list-variables: use 0 or --no-values
        self.runCmd("-stack-list-variables 0")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"c\"},{arg=\"1\",name=\"d\"},{name=\"a\"},{name=\"b\"}\]")
        self.runCmd("-stack-list-variables --no-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"c\"},{arg=\"1\",name=\"d\"},{name=\"a\"},{name=\"b\"}\]")

        # Test -stack-list-variables: use 1 or --all-values
        self.runCmd("-stack-list-variables 1")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"c\",value=\"30\"},{arg=\"1\",name=\"d\",value=\"40\"},{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")
        self.runCmd("-stack-list-variables --all-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"c\",value=\"30\"},{arg=\"1\",name=\"d\",value=\"40\"},{name=\"a\",value=\"10\"},{name=\"b\",value=\"20\"}\]")

        # Test -stack-list-variables: use 2 or --simple-values
        self.runCmd("-stack-list-variables 2")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"c\",type=\"int\",value=\"30\"},{arg=\"1\",name=\"d\",type=\"int\",value=\"40\"},{name=\"a\",type=\"int\",value=\"10\"},{name=\"b\",type=\"int\",value=\"20\"}\]")
        self.runCmd("-stack-list-variables --simple-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"c\",type=\"int\",value=\"30\"},{arg=\"1\",name=\"d\",type=\"int\",value=\"40\"},{name=\"a\",type=\"int\",value=\"10\"},{name=\"b\",type=\"int\",value=\"20\"}\]")
        
        # Test struct local variable:
        # Run to BP_local_struct_test
        line = line_number('main.cpp', '// BP_local_struct_test_with_args')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"3\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-variables: use 0 or --no-values
        self.runCmd("-stack-list-variables 0")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"var_e\"},{name=\"var_c\"}\]")
        self.runCmd("-stack-list-variables --no-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"var_e\"},{name=\"var_c\"}\]")

        # Test -stack-list-variables: use 1 or --all-values
        self.runCmd("-stack-list-variables 1")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"var_e\",value=\"{var_a = 20, var_b = 98 'b', inner_ = {var_d = 40}}\"},{name=\"var_c\",value=\"{var_a = 10, var_b = 97 'a', inner_ = {var_d = 30}}\"}\]")
        self.runCmd("-stack-list-variables --all-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"var_e\",value=\"{var_a = 20, var_b = 98 'b', inner_ = {var_d = 40}}\"},{name=\"var_c\",value=\"{var_a = 10, var_b = 97 'a', inner_ = {var_d = 30}}\"}\]")

        # Test -stack-list-variables: use 2 or --simple-values
        self.runCmd("-stack-list-variables 2")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"var_e\",type=\"my_type\"},{name=\"var_c\",type=\"my_type\"}\]")
        self.runCmd("-stack-list-variables --simple-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"var_e\",type=\"my_type\"},{name=\"var_c\",type=\"my_type\"}\]")
        
        # Test array local variable:
        # Run to BP_local_array_test
        line = line_number('main.cpp', '// BP_local_array_test_with_args')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"4\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-variables: use 0 or --no-values
        self.runCmd("-stack-list-variables 0")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"other_array\"},{name=\"array\"}\]")
        self.runCmd("-stack-list-variables --no-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"other_array\"},{name=\"array\"}\]")

        # Test -stack-list-variables: use 1 or --all-values
        self.runCmd("-stack-list-variables 1")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"other_array\",value=\".*?\"},{name=\"array\",value=\"{\[0\] = 100, \[1\] = 200, \[2\] = 300}\"}\]")
        self.runCmd("-stack-list-variables --all-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"other_array\",value=\".*?\"},{name=\"array\",value=\"{\[0\] = 100, \[1\] = 200, \[2\] = 300}\"}\]")

        # Test -stack-list-variables: use 2 or --simple-values
        self.runCmd("-stack-list-variables 2")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"other_array\",type=\"int \*\",value=\".*?\"},{name=\"array\",type=\"int \[3\]\"}\]")
        self.runCmd("-stack-list-variables --simple-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"other_array\",type=\"int \*\",value=\".*?\"},{name=\"array\",type=\"int \[3\]\"}\]")
        
        # Test pointers as local variable:
        # Run to BP_local_pointer_test
        line = line_number('main.cpp', '// BP_local_pointer_test_with_args')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"5\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        
        # Test -stack-list-variables: use 0 or --no-values
        self.runCmd("-stack-list-variables 0")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"arg_str\"},{arg=\"1\",name=\"arg_ptr\"},{name=\"test_str\"},{name=\"var_e\"},{name=\"ptr\"}\]")
        self.runCmd("-stack-list-variables --no-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"arg_str\"},{arg=\"1\",name=\"arg_ptr\"},{name=\"test_str\"},{name=\"var_e\"},{name=\"ptr\"}\]")

        # Test -stack-list-variables: use 1 or --all-values
        self.runCmd("-stack-list-variables 1")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"arg_str\",value=\".*?String.*?\"},{arg=\"1\",name=\"arg_ptr\",value=\".*?\"},{name=\"test_str\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*?\"}\]")
        self.runCmd("-stack-list-variables --all-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"arg_str\",value=\".*?String.*?\"},{arg=\"1\",name=\"arg_ptr\",value=\".*?\"},{name=\"test_str\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",value=\"24\"},{name=\"ptr\",value=\".*?\"}\]")

        # Test -stack-list-variables: use 2 or --simple-values
        self.runCmd("-stack-list-variables 2")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"arg_str\",type=\"const char \*\",value=\".*?String.*?\"},{arg=\"1\",name=\"arg_ptr\",type=\"int \*\",value=\".*?\"},{name=\"test_str\",type=\"const char \*\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",type=\"int\",value=\"24\"},{name=\"ptr\",type=\"int \*\",value=\".*?\"}\]")
        self.runCmd("-stack-list-variables --simple-values")
        self.expect("\^done,variables=\[{arg=\"1\",name=\"arg_str\",type=\"const char \*\",value=\".*?String.*?\"},{arg=\"1\",name=\"arg_ptr\",type=\"int \*\",value=\".*?\"},{name=\"test_str\",type=\"const char \*\",value=\".*?Rakaposhi.*?\"},{name=\"var_e\",type=\"int\",value=\"24\"},{name=\"ptr\",type=\"int \*\",value=\".*?\"}\]")
        
    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_info_depth(self):
        """Test that 'lldb-mi --interpreter' can shows depth of the stack."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -stack-info-depth works
        # (and that max-depth is optional)
        self.runCmd("-stack-info-depth")
        self.expect("\^done,depth=\"[1-9]\"")

        # Test that max-depth restricts check of stack depth
        #FIXME: max-depth argument is ignored
        self.runCmd("-stack-info-depth 1")
        #self.expect("\^done,depth=\"1\"")

        # Test that invalid max-depth argument is handled
        #FIXME: max-depth argument is ignored
        self.runCmd("-stack-info-depth -1")
        #self.expect("\^error")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipUnlessDarwin
    def test_lldbmi_stack_info_frame(self):
        """Test that 'lldb-mi --interpreter' can show information about current frame."""

        self.spawnLldbMi(args = None)

        # Test that -stack-info-frame fails when program isn't running
        self.runCmd("-stack-info-frame")
        self.expect("\^error,msg=\"Command 'stack-info-frame'\. Invalid process during debug session\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -stack-info-frame works when program was stopped on BP
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\"0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\"\}")

        # Select frame #1
        self.runCmd("-stack-select-frame 1")
        self.expect("\^done")

        # Test that -stack-info-frame works when specified frame was selected
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"1\",addr=\"0x[0-9a-f]+\",func=\".+?\",file=\"\?\?\",fullname=\"\?\?\",line=\"-1\"\}")

        # Test that -stack-info-frame fails when an argument is specified
        #FIXME: unknown argument is ignored
        self.runCmd("-stack-info-frame unknown_arg")
        #self.expect("\^error")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_list_frames(self):
        """Test that 'lldb-mi --interpreter' can lists the frames on the stack."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test stack frame: get frame #0 info
        self.runCmd("-stack-list-frames 0 0")
        self.expect("\^done,stack=\[frame=\{level=\"0\",addr=\"0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\"\}\]")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_stack_select_frame(self):
        """Test that 'lldb-mi --interpreter' can choose current frame."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -stack-select-frame requires 1 mandatory argument
        self.runCmd("-stack-select-frame")
        self.expect("\^error,msg=\"Command 'stack-select-frame'\. Command Args\. Validation failed. Mandatory args not found: frame_id\"")

        # Test that -stack-select-frame fails on invalid frame number
        self.runCmd("-stack-select-frame 99")
        self.expect("\^error,msg=\"Command 'stack-select-frame'\. Frame ID invalid\"")

        # Test that current frame is #0
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\"0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\"\}")

        # Test that -stack-select-frame can select the selected frame
        self.runCmd("-stack-select-frame 0")
        self.expect("\^done")

        # Test that current frame is still #0
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\"0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\"\}")

        # Test that -stack-select-frame can select frame #1 (parent frame)
        self.runCmd("-stack-select-frame 1")
        self.expect("\^done")

        # Test that current frame is #1
        # Note that message is different in Darwin and Linux:
        # Darwin: "^done,frame={level=\"1\",addr=\"0x[0-9a-f]+\",func=\"start\",file=\"??\",fullname=\"??\",line=\"-1\"}"
        # Linux:  "^done,frame={level=\"1\",addr=\"0x[0-9a-f]+\",func=\".+\",file=\".+\",fullname=\".+\",line=\"\d+\"}"
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"1\",addr=\"0x[0-9a-f]+\",func=\".+?\",file=\".+?\",fullname=\".+?\",line=\"(-1|\d+)\"\}")

        # Test that -stack-select-frame can select frame #0 (child frame)
        self.runCmd("-stack-select-frame 0")
        self.expect("\^done")

        # Test that current frame is #0 and it has the same information
        self.runCmd("-stack-info-frame")
        self.expect("\^done,frame=\{level=\"0\",addr=\"0x[0-9a-f]+\",func=\"main\",file=\"main\.cpp\",fullname=\".+?main\.cpp\",line=\"\d+\"\}")
