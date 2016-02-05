#coding=utf8
"""
Test lldb-mi -gdb-set and -gdb-show commands for 'print option-name'.
"""

from __future__ import print_function



import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class MiGdbSetShowTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    # evaluates array when char-array-as-string is off
    def eval_and_check_array(self, var, typ, length):
        self.runCmd("-var-create - * %s" % var)
        self.expect('\^done,name="var\d+",numchild="%d",value="\[%d\]",type="%s \[%d\]",thread-id="1",has_more="0"' % (length, length, typ, length))

    # evaluates any type which can be represented as string of characters
    def eval_and_match_string(self, var, value, typ):
        value=value.replace("\\", "\\\\").replace("\"", "\\\"")
        self.runCmd("-var-create - * " + var)
        self.expect('\^done,name="var\d+",numchild="[0-9]+",value="%s",type="%s",thread-id="1",has_more="0"' % (value, typ))

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_gdb_set_show_print_char_array_as_string(self):
        """Test that 'lldb-mi --interpreter' can print array of chars as string."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_gdb_set_show_print_char_array_as_string_test
        line = line_number('main.cpp', '// BP_gdb_set_show_print_char_array_as_string_test')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that default print char-array-as-string value is "off"
        self.runCmd("-gdb-show print char-array-as-string")
        self.expect("\^done,value=\"off\"")

        # Test that a char* is expanded to string when print char-array-as-string is "off"
        self.eval_and_match_string("cp", r'0x[0-9a-f]+ \"\\t\\\"hello\\\"\\n\"', r'const char \*')

        # Test that a char[] isn't expanded to string when print char-array-as-string is "off"
        self.eval_and_check_array("ca", "const char", 10);

        # Test that a char16_t* is expanded to string when print char-array-as-string is "off"
        self.eval_and_match_string("u16p", r'0x[0-9a-f]+ u\"\\t\\\"hello\\\"\\n\"', r'const char16_t \*')

        # Test that a char16_t[] isn't expanded to string when print char-array-as-string is "off"
        self.eval_and_check_array("u16a", "const char16_t", 10);

        # Test that a char32_t* is expanded to string when print char-array-as-string is "off"
        self.eval_and_match_string("u32p", r'0x[0-9a-f]+ U\"\\t\\\"hello\\\"\\n\"', r'const char32_t \*')

        # Test that a char32_t[] isn't expanded to string when print char-array-as-string is "off"
        self.eval_and_check_array("u32a", "const char32_t", 10);

        # Test that -gdb-set can set print char-array-as-string flag
        self.runCmd("-gdb-set print char-array-as-string on")
        self.expect("\^done")
        self.runCmd("-gdb-set print char-array-as-string 1")
        self.expect("\^done")
        self.runCmd("-gdb-show print char-array-as-string")
        self.expect("\^done,value=\"on\"")

        # Test that a char* with escape chars is expanded to string when print char-array-as-string is "on"
        self.eval_and_match_string("cp", r'0x[0-9a-f]+ \"\\t\\\"hello\\\"\\n\"', r'const char \*')
        
        # Test that a char[] with escape chars is expanded to string when print char-array-as-string is "on"
        self.eval_and_match_string("ca", r'\"\\t\\\"hello\\\"\\n\"', r'const char \[10\]')
        
        # Test that a char16_t* with escape chars is expanded to string when print char-array-as-string is "on"
        self.eval_and_match_string("u16p", r'0x[0-9a-f]+ u\"\\t\\\"hello\\\"\\n\"', r'const char16_t \*')
        
        # Test that a char16_t[] with escape chars is expanded to string when print char-array-as-string is "on"
        self.eval_and_match_string("u16a", r'u\"\\t\\\"hello\\\"\\n\"', r'const char16_t \[10\]')
        
        # Test that a char32_t* with escape chars is expanded to string when print char-array-as-string is "on"
        self.eval_and_match_string("u32p", r'0x[0-9a-f]+ U\"\\t\\\"hello\\\"\\n\"', r'const char32_t \*')
        
        # Test that a char32_t[] with escape chars is expanded to string when print char-array-as-string is "on"
        self.eval_and_match_string("u32a", r'U\"\\t\\\"hello\\\"\\n\"', r'const char32_t \[10\]')

        # Test russian unicode strings
        self.eval_and_match_string("u16p_rus", r'0x[0-9a-f]+ u\"\\\\Аламо-сквер\"', r'const char16_t \*')
        self.eval_and_match_string("u16a_rus", r'u\"\\\\Бейвью\"', r'const char16_t \[8\]')
        self.eval_and_match_string("u32p_rus", r'0x[0-9a-f]+ U\"\\\\Чайнатаун\"', r'const char32_t \*')
        self.eval_and_match_string("u32a_rus", r'U\"\\\\Догпатч\"', r'const char32_t \[9\]')

        # Test that -gdb-set print char-array-as-string fails if "on"/"off" isn't specified
        self.runCmd("-gdb-set print char-array-as-string")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

        # Test that -gdb-set print char-array-as-string fails when option is unknown
        self.runCmd("-gdb-set print char-array-as-string unknown")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi working on Windows
    @expectedFailureGcc("https://llvm.org/bugs/show_bug.cgi?id=23357")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_gdb_set_show_print_expand_aggregates(self):
        """Test that 'lldb-mi --interpreter' can expand aggregates everywhere."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_gdb_set_show_print_expand_aggregates
        line = line_number('main.cpp', '// BP_gdb_set_show_print_expand_aggregates')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that default print expand-aggregates value is "off"
        self.runCmd("-gdb-show print expand-aggregates")
        self.expect("\^done,value=\"off\"")

        # Test that composite type isn't expanded when print expand-aggregates is "off"
        self.runCmd("-var-create var1 * complx")
        self.expect("\^done,name=\"var1\",numchild=\"3\",value=\"{\.\.\.}\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"")

        # Test that composite type[] isn't expanded when print expand-aggregates is "off"
        self.eval_and_check_array("complx_array", "complex_type", 2)

        # Test that a struct with a char first element is not formatted as a string
        self.runCmd("-var-create - * &nstr")
        self.expect("\^done,name=\"var\d+\",numchild=\"2\",value=\"0x[0-9a-f]+\",type=\"not_str \*\",thread-id=\"1\",has_more=\"0\"")

        # Test that -gdb-set can set print expand-aggregates flag
        self.runCmd("-gdb-set print expand-aggregates on")
        self.expect("\^done")
        self.runCmd("-gdb-set print expand-aggregates 1")
        self.expect("\^done")
        self.runCmd("-gdb-show print expand-aggregates")
        self.expect("\^done,value=\"on\"")

        # Test that composite type is expanded when print expand-aggregates is "on"
        self.runCmd("-var-create var3 * complx")
        self.expect("\^done,name=\"var3\",numchild=\"3\",value=\"{i = 3, inner = {l = 3}, complex_ptr = 0x[0-9a-f]+}\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"")

        # Test that composite type[] is expanded when print expand-aggregates is "on"
        self.runCmd("-var-create var4 * complx_array")
        self.expect("\^done,name=\"var4\",numchild=\"2\",value=\"{\[0\] = {i = 4, inner = {l = 4}, complex_ptr = 0x[0-9a-f]+}, \[1\] = {i = 5, inner = {l = 5}, complex_ptr = 0x[0-9a-f]+}}\",type=\"complex_type \[2\]\",thread-id=\"1\",has_more=\"0\"")

        # Test that -gdb-set print expand-aggregates fails if "on"/"off" isn't specified
        self.runCmd("-gdb-set print expand-aggregates")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

        # Test that -gdb-set print expand-aggregates fails when option is unknown
        self.runCmd("-gdb-set print expand-aggregates unknown")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi working on Windows
    @expectedFailureGcc("https://llvm.org/bugs/show_bug.cgi?id=23357")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_gdb_set_show_print_aggregate_field_names(self):
        """Test that 'lldb-mi --interpreter' can expand aggregates everywhere."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_gdb_set_show_print_aggregate_field_names
        line = line_number('main.cpp', '// BP_gdb_set_show_print_aggregate_field_names')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that default print aggregatep-field-names value is "on"
        self.runCmd("-gdb-show print aggregate-field-names")
        self.expect("\^done,value=\"on\"")

        # Set print expand-aggregates flag to "on"
        self.runCmd("-gdb-set print expand-aggregates on")
        self.expect("\^done")

        # Test that composite type is expanded with field name when print aggregate-field-names is "on"
        self.runCmd("-var-create var1 * complx")
        self.expect("\^done,name=\"var1\",numchild=\"3\",value=\"{i = 3, inner = {l = 3}, complex_ptr = 0x[0-9a-f]+}\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"")

        # Test that composite type[] is expanded with field name when print aggregate-field-names is "on"
        self.runCmd("-var-create var2 * complx_array")
        self.expect("\^done,name=\"var2\",numchild=\"2\",value=\"{\[0\] = {i = 4, inner = {l = 4}, complex_ptr = 0x[0-9a-f]+}, \[1\] = {i = 5, inner = {l = 5}, complex_ptr = 0x[0-9a-f]+}}\",type=\"complex_type \[2\]\",thread-id=\"1\",has_more=\"0\"")

        # Test that -gdb-set can set print aggregate-field-names flag
        self.runCmd("-gdb-set print aggregate-field-names off")
        self.expect("\^done")
        self.runCmd("-gdb-set print aggregate-field-names 0")
        self.expect("\^done")
        self.runCmd("-gdb-show print aggregate-field-names")
        self.expect("\^done,value=\"off\"")

        # Test that composite type is expanded without field name when print aggregate-field-names is "off"
        self.runCmd("-var-create var3 * complx")
        self.expect("\^done,name=\"var3\",numchild=\"3\",value=\"{3,\{3\},0x[0-9a-f]+}\",type=\"complex_type\",thread-id=\"1\",has_more=\"0\"")

        # Test that composite type[] is expanded without field name when print aggregate-field-names is "off"
        self.runCmd("-var-create var4 * complx_array")
        self.expect("\^done,name=\"var4\",numchild=\"2\",value=\"{{4,\{4\},0x[0-9a-f]+},{5,\{5\},0x[0-9a-f]+}}\",type=\"complex_type \[2\]\",thread-id=\"1\",has_more=\"0\"")

        # Test that -gdb-set print aggregate-field-names fails if "on"/"off" isn't specified
        self.runCmd("-gdb-set print aggregate-field-names")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")

        # Test that -gdb-set print aggregate-field-names fails when option is unknown
        self.runCmd("-gdb-set print aggregate-field-names unknown")
        self.expect("\^error,msg=\"The request ''print' expects option-name and \"on\" or \"off\"' failed.\"")
