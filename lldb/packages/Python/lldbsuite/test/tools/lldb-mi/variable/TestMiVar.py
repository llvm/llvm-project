"""
Test lldb-mi -var-xxx commands.
"""

from __future__ import print_function



import lldbmi_testcase
from lldbsuite.test.lldbtest import *

class MiVarTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_eval(self):
        """Test that 'lldb-mi --interpreter' works for evaluating."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to program return
        line = line_number('main.cpp', '// BP_return')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Print non-existant variable
        self.runCmd("-var-create var1 * undef")
        self.expect("\^error,msg=\"error: error: use of undeclared identifier \'undef\'\\\\nerror: 1 errors parsing expression\\\\n\"")
        self.runCmd("-data-evaluate-expression undef")
        self.expect("\^error,msg=\"Could not evaluate expression\"")

        # Print global "g_MyVar", modify, delete and create again
        self.runCmd("-data-evaluate-expression g_MyVar")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-create var2 * g_MyVar")
        self.expect("\^done,name=\"var2\",numchild=\"0\",value=\"3\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var2")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-show-attributes var2")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-var-list-children var2")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")
        # Ensure -var-list-children also works with quotes
        self.runCmd("-var-list-children \"var2\"")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")
        self.runCmd("-data-evaluate-expression \"g_MyVar=30\"")
        self.expect("\^done,value=\"30\"")
        self.runCmd("-var-update --all-values var2")
        #self.expect("\^done,changelist=\[\{name=\"var2\",value=\"30\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"\}\]") #FIXME -var-update doesn't work
        self.runCmd("-var-delete var2")
        self.expect("\^done")
        self.runCmd("-var-create var2 * g_MyVar")
        self.expect("\^done,name=\"var2\",numchild=\"0\",value=\"30\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Print static "s_MyVar", modify, delete and create again
        self.runCmd("-data-evaluate-expression s_MyVar")
        self.expect("\^done,value=\"30\"")
        self.runCmd("-var-create var3 * s_MyVar")
        self.expect("\^done,name=\"var3\",numchild=\"0\",value=\"30\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var3")
        self.expect("\^done,value=\"30\"")
        self.runCmd("-var-show-attributes var3")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-var-list-children var3")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")
        self.runCmd("-data-evaluate-expression \"s_MyVar=3\"")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-update --all-values var3")
        #self.expect("\^done,changelist=\[\{name=\"var3\",value=\"3\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"\}\]") #FIXME -var-update doesn't work
        self.runCmd("-var-delete var3")
        self.expect("\^done")
        self.runCmd("-var-create var3 * s_MyVar")
        self.expect("\^done,name=\"var3\",numchild=\"0\",value=\"3\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Print local "b", modify, delete and create again
        self.runCmd("-data-evaluate-expression b")
        self.expect("\^done,value=\"20\"")
        self.runCmd("-var-create var4 * b")
        self.expect("\^done,name=\"var4\",numchild=\"0\",value=\"20\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var4")
        self.expect("\^done,value=\"20\"")
        self.runCmd("-var-show-attributes var4")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-var-list-children var4")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")
        self.runCmd("-data-evaluate-expression \"b=2\"")
        self.expect("\^done,value=\"2\"")
        self.runCmd("-var-update --all-values var4")
        #self.expect("\^done,changelist=\[\{name=\"var4\",value=\"2\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"\}\]") #FIXME -var-update doesn't work
        self.runCmd("-var-delete var4")
        self.expect("\^done")
        self.runCmd("-var-create var4 * b")
        self.expect("\^done,name=\"var4\",numchild=\"0\",value=\"2\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Print temp "a + b"
        self.runCmd("-data-evaluate-expression \"a + b\"")
        self.expect("\^done,value=\"12\"")
        self.runCmd("-var-create var5 * \"a + b\"")
        self.expect("\^done,name=\"var5\",numchild=\"0\",value=\"12\",type=\"int\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-evaluate-expression var5")
        self.expect("\^done,value=\"12\"")
        self.runCmd("-var-show-attributes var5")
        self.expect("\^done,status=\"editable\"") #FIXME editable or not?
        self.runCmd("-var-list-children var5")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")

        # Print argument "argv[0]"
        self.runCmd("-data-evaluate-expression \"argv[0]\"")
        self.expect("\^done,value=\"0x[0-9a-f]+ \\\\\\\".*?%s\\\\\\\"\"" % self.myexe)
        self.runCmd("-var-create var6 * \"argv[0]\"")
        self.expect("\^done,name=\"var6\",numchild=\"1\",value=\"0x[0-9a-f]+ \\\\\\\".*?%s\\\\\\\"\",type=\"const char \*\",thread-id=\"1\",has_more=\"0\"" % self.myexe)
        self.runCmd("-var-evaluate-expression var6")
        self.expect("\^done,value=\"0x[0-9a-f]+ \\\\\\\".*?%s\\\\\\\"\"" % self.myexe)
        self.runCmd("-var-show-attributes var6")
        self.expect("\^done,status=\"editable\"")
        self.runCmd("-var-list-children --all-values var6")
        # FIXME: The name below is not correct. It should be "var.*argv[0]".
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var6\.\*\$[0-9]+\",exp=\"\*\$[0-9]+\",numchild=\"0\",type=\"const char\",thread-id=\"4294967295\",value=\"47 '/'\",has_more=\"0\"\}\],has_more=\"0\"") #FIXME -var-list-children shows invalid thread-id

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_var_update(self):
        """Test that 'lldb-mi --interpreter' works for -var-update."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_var_update_test_init
        line = line_number('main.cpp', '// BP_var_update_test_init')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Setup variables
        self.runCmd("-var-create var_l * l")
        self.expect("\^done,name=\"var_l\",numchild=\"0\",value=\"1\",type=\"long\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-create var_complx * complx")
        self.expect("\^done,name=\"var_complx\",numchild=\"3\",value=\"\{\.\.\.\}\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-create var_complx_array * complx_array")
        self.expect("\^done,name=\"var_complx_array\",numchild=\"2\",value=\"\[2\]\",type=\"complex_type \[2\]\",thread-id=\"1\",has_more=\"0\"")

        # Go to BP_var_update_test_l
        line = line_number('main.cpp', '// BP_var_update_test_l')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that var_l was updated
        self.runCmd("-var-update --all-values var_l")
        self.expect("\^done,changelist=\[\{name=\"var_l\",value=\"0\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"\}\]")

        # Go to BP_var_update_test_complx
        line = line_number('main.cpp', '// BP_var_update_test_complx')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"3\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that var_complx was updated
        self.runCmd("-var-update --all-values var_complx")
        self.expect("\^done,changelist=\[\{name=\"var_complx\",value=\"\{\.\.\.\}\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"\}\]")

        # Go to BP_var_update_test_complx_array
        line = line_number('main.cpp', '// BP_var_update_test_complx_array')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"4\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that var_complex_array was updated
        self.runCmd("-var-update --all-values var_complx_array")
        self.expect("\^done,changelist=\[\{name=\"var_complx_array\",value=\"\[2\]\",in_scope=\"true\",type_changed=\"false\",has_more=\"0\"\}\]")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_var_create_register(self):
        """Test that 'lldb-mi --interpreter' works for -var-create $regname."""

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

        # Find name of register 0
        self.runCmd("-data-list-register-names 0")
        self.expect("\^done,register-names=\[\".+?\"\]")
        register_name = self.child.after.split("\"")[1]

        # Create variable for register 0
        # Note that message is different in Darwin and Linux:
        # Darwin: "^done,name=\"var_reg\",numchild=\"0\",value=\"0x[0-9a-f]+\",type=\"unsigned long\",thread-id=\"1\",has_more=\"0\"
        # Linux:  "^done,name=\"var_reg\",numchild=\"0\",value=\"0x[0-9a-f]+\",type=\"unsigned int\",thread-id=\"1\",has_more=\"0\"
        self.runCmd("-var-create var_reg * $%s" % register_name)
        self.expect("\^done,name=\"var_reg\",numchild=\"0\",value=\"0x[0-9a-f]+\",type=\"unsigned (long|int)\",thread-id=\"1\",has_more=\"0\"")

        # Assign value to variable
        self.runCmd("-var-assign var_reg \"6\"")
        #FIXME: the output has different format for 32bit and 64bit values
        self.expect("\^done,value=\"0x0*?6\"")

        # Assert register 0 updated
        self.runCmd("-data-list-register-values d 0")
        self.expect("\^done,register-values=\[{number=\"0\",value=\"6\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_var_list_children(self):
        """Test that 'lldb-mi --interpreter' works for -var-list-children."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_var_list_children_test
        line = line_number('main.cpp', '// BP_var_list_children_test')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Create variable
        self.runCmd("-var-create var_complx * complx")
        self.expect("\^done,name=\"var_complx\",numchild=\"3\",value=\"\{\.\.\.\}\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-create var_complx_array * complx_array")
        self.expect("\^done,name=\"var_complx_array\",numchild=\"2\",value=\"\[2\]\",type=\"complex_type \[2\]\",thread-id=\"1\",has_more=\"0\"")
        self.runCmd("-var-create var_pcomplx * pcomplx")
        self.expect("\^done,name=\"var_pcomplx\",numchild=\"2\",value=\"\{\.\.\.\}\",type=\"pcomplex_type\",thread-id=\"1\",has_more=\"0\"")

        # Test that -var-evaluate-expression can evaluate the children of created varobj
        self.runCmd("-var-list-children var_complx")
        self.runCmd("-var-evaluate-expression var_complx.i")
        self.expect("\^done,value=\"3\"")
        self.runCmd("-var-list-children var_complx_array")
        self.runCmd("-var-evaluate-expression var_complx_array.[0]")
        self.expect("\^done,value=\"\{...\}\"")
        self.runCmd("-var-list-children var_pcomplx")
        self.runCmd("-var-evaluate-expression var_pcomplx.complex_type")
        self.expect("\^done,value=\"\{...\}\"")

        # Test that -var-list-children lists empty children if range is empty
        # (and that print-values is optional)
        self.runCmd("-var-list-children var_complx 0 0")
        self.expect("\^done,numchild=\"0\",has_more=\"1\"")
        self.runCmd("-var-list-children var_complx 99 0")
        self.expect("\^done,numchild=\"0\",has_more=\"1\"")
        self.runCmd("-var-list-children var_complx 99 3")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")

        # Test that -var-list-children lists all children with their values
        # (and that from and to are optional)
        self.runCmd("-var-list-children --all-values var_complx")
        self.expect("\^done,numchild=\"3\",children=\[child=\{name=\"var_complx\.i\",exp=\"i\",numchild=\"0\",type=\"int\",thread-id=\"1\",value=\"3\",has_more=\"0\"\},child=\{name=\"var_complx\.inner\",exp=\"inner\",numchild=\"1\",type=\"complex_type::\(anonymous struct\)\",thread-id=\"1\",value=\"\{\.\.\.\}\",has_more=\"0\"\},child=\{name=\"var_complx\.complex_ptr\",exp=\"complex_ptr\",numchild=\"3\",type=\"complex_type \*\",thread-id=\"1\",value=\"0x[0-9a-f]+\",has_more=\"0\"\}\],has_more=\"0\"")
        self.runCmd("-var-list-children --simple-values var_complx_array")
        self.expect("\^done,numchild=\"2\",children=\[child=\{name=\"var_complx_array\.\[0\]\",exp=\"\[0\]\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"\},child=\{name=\"var_complx_array\.\[1\]\",exp=\"\[1\]\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"0\"")
        self.runCmd("-var-list-children 0 var_pcomplx")
        self.expect("\^done,numchild=\"2\",children=\[child=\{name=\"var_pcomplx\.complex_type\",exp=\"complex_type\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"\},child={name=\"var_pcomplx\.complx\",exp=\"complx\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"0\"")

        # Test that -var-list-children lists children without values
        self.runCmd("-var-list-children 0 var_complx 0 1")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx\.i\",exp=\"i\",numchild=\"0\",type=\"int\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"1\"")
        self.runCmd("-var-list-children --no-values var_complx 0 1")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx\.i\",exp=\"i\",numchild=\"0\",type=\"int\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"1\"")
        self.runCmd("-var-list-children --no-values var_complx_array 0 1")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx_array\.\[0\]\",exp=\"\[0\]\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"1\"")
        self.runCmd("-var-list-children --no-values var_pcomplx 0 1")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_pcomplx\.complex_type\",exp=\"complex_type\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"1\"")

        # Test that -var-list-children lists children with all values
        self.runCmd("-var-list-children 1 var_complx 1 2")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx\.inner\",exp=\"inner\",numchild=\"1\",type=\"complex_type::\(anonymous struct\)\",thread-id=\"1\",value=\"\{\.\.\.\}\",has_more=\"0\"\}\],has_more=\"1\"")
        self.runCmd("-var-list-children --all-values var_complx 1 2")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx\.inner\",exp=\"inner\",numchild=\"1\",type=\"complex_type::\(anonymous struct\)\",thread-id=\"1\",value=\"\{\.\.\.\}\",has_more=\"0\"\}\],has_more=\"1\"")
        self.runCmd("-var-list-children --all-values var_complx_array 1 2")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx_array\.\[1\]\",exp=\"\[1\]\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",value=\"\{\.\.\.\}\",has_more=\"0\"\}\],has_more=\"0\"")
        self.runCmd("-var-list-children --all-values var_pcomplx 1 2")
        self.expect("\^done,numchild=\"1\",children=\[child={name=\"var_pcomplx\.complx\",exp=\"complx\",numchild=\"3\",type=\"complex_type\",thread-id=\"1\",value=\"\{\.\.\.\}\",has_more=\"0\"\}\],has_more=\"0\"")

        # Test that -var-list-children lists children with simple values
        self.runCmd("-var-list-children 2 var_complx 2 4")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx\.complex_ptr\",exp=\"complex_ptr\",numchild=\"3\",type=\"complex_type \*\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"0\"")
        self.runCmd("-var-list-children --simple-values var_complx 2 4")
        self.expect("\^done,numchild=\"1\",children=\[child=\{name=\"var_complx\.complex_ptr\",exp=\"complex_ptr\",numchild=\"3\",type=\"complex_type \*\",thread-id=\"1\",has_more=\"0\"\}\],has_more=\"0\"")
        self.runCmd("-var-list-children --simple-values var_complx_array 2 4")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")
        self.runCmd("-var-list-children --simple-values var_pcomplx 2 4")
        self.expect("\^done,numchild=\"0\",has_more=\"0\"")

        # Test that an invalid from is handled
        # FIXME: -1 is treated as unsigned int
        self.runCmd("-var-list-children 0 var_complx -1 0")
        #self.expect("\^error,msg=\"Command 'var-list-children'\. Variable children range invalid\"")

        # Test that an invalid to is handled
        # FIXME: -1 is treated as unsigned int
        self.runCmd("-var-list-children 0 var_complx 0 -1")
        #self.expect("\^error,msg=\"Command 'var-list-children'\. Variable children range invalid\"")

        # Test that a missing low-frame or high-frame is handled
        self.runCmd("-var-list-children 0 var_complx 0")
        self.expect("\^error,msg=\"Command 'var-list-children'. Variable children range invalid\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_var_create_for_stl_types(self):
        """Test that 'lldb-mi --interpreter' print summary for STL types."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_gdb_set_show_print_char_array_as_string_test
        line = line_number('main.cpp', '// BP_cpp_stl_types_test')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test for std::string
        self.runCmd("-var-create - * std_string")
        self.expect('\^done,name="var\d+",numchild="[0-9]+",value="\\\\"hello\\\\"",type="std::[\S]*?string",thread-id="1",has_more="0"')
 
    @skipIfWindows #llvm.org/pr24452: Get lldb-mi working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_var_create_for_unnamed_objects(self):
        """Test that 'lldb-mi --interpreter' can expand unnamed structures and unions."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to breakpoint
        line = line_number('main.cpp', '// BP_unnamed_objects_test')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Evaluate struct_with_unions type and its children
        self.runCmd("-var-create v0 * swu")
        self.expect('\^done,name="v0",numchild="2",value="\{\.\.\.\}",type="struct_with_unions",thread-id="1",has_more="0"')
       
        self.runCmd("-var-list-children v0")
        
        # inspect the first unnamed union
        self.runCmd("-var-list-children v0.$0")
        self.runCmd("-var-evaluate-expression v0.$0.u_i")
        self.expect('\^done,value="1"')
        
        # inspect the second unnamed union
        self.runCmd("-var-list-children v0.$1")
        self.runCmd("-var-evaluate-expression v0.$1.u1")
        self.expect('\^done,value="-1"')
        # inspect unnamed structure
        self.runCmd("-var-list-children v0.$1.$1")
        self.runCmd("-var-evaluate-expression v0.$1.$1.s1")
        self.expect('\^done,value="-1"')

