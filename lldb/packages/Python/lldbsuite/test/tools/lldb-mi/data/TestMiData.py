"""
Test lldb-mi -data-xxx commands.
"""

from __future__ import print_function


import unittest2
import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiDataTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_disassemble(self):
        """Test that 'lldb-mi --interpreter' works for -data-disassemble."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Get an address for disassembling: use main
        self.runCmd("-data-evaluate-expression main")
        self.expect(
            "\^done,value=\"0x[0-9a-f]+ \(a.out`main at main.cpp:[0-9]+\)\"")
        addr = int(self.child.after.split("\"")[1].split(" ")[0], 16)

        # Test -data-disassemble: try to disassemble some address
        self.runCmd(
            "-data-disassemble -s %#x -e %#x -- 0" %
            (addr, addr + 0x10))
        self.expect(
            "\^done,asm_insns=\[{address=\"0x0*%x\",func-name=\"main\",offset=\"0\",size=\"[1-9]+\",inst=\".+?\"}," %
            addr)

        # Test -data-disassemble without "--"
        self.runCmd("-data-disassemble -s %#x -e %#x 0" % (addr, addr + 0x10))
        self.expect(
            "\^done,asm_insns=\[{address=\"0x0*%x\",func-name=\"main\",offset=\"0\",size=\"[1-9]+\",inst=\".+?\"}," %
            addr)

        # Run to hello_world
        self.runCmd("-break-insert -f hello_world")
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Get an address for disassembling: use hello_world
        self.runCmd("-data-evaluate-expression hello_world")
        self.expect(
            "\^done,value=\"0x[0-9a-f]+ \(a.out`hello_world\(\) at main.cpp:[0-9]+\)\"")
        addr = int(self.child.after.split("\"")[1].split(" ")[0], 16)

        # Test -data-disassemble: try to disassemble some address
        self.runCmd(
            "-data-disassemble -s %#x -e %#x -- 0" %
            (addr, addr + 0x10))

        # This matches a line similar to:
        # Darwin: {address="0x0000000100000f18",func-name="hello_world()",offset="8",size="7",inst="leaq 0x65(%rip), %rdi; \"Hello, World!\\n\""},
        # Linux:  {address="0x0000000000400642",func-name="hello_world()",offset="18",size="5",inst="callq 0x4004d0; symbol stub for: printf"}
        # To match the escaped characters in the ouptut, we must use four backslashes per matches backslash
        # See https://docs.python.org/2/howto/regex.html#the-backslash-plague

        # The MIPS and PPC64le disassemblers never print stub name
        if self.isMIPS() or self.isPPC64le():
            self.expect(["{address=\"0x[0-9a-f]+\",func-name=\"hello_world\(\)\",offset=\"[0-9]+\",size=\"[0-9]+\",inst=\".+?; \\\\\"Hello, World!\\\\\\\\n\\\\\"\"}",
                     "{address=\"0x[0-9a-f]+\",func-name=\"hello_world\(\)\",offset=\"[0-9]+\",size=\"[0-9]+\",inst=\".+?\"}"])
        else:
            self.expect(["{address=\"0x[0-9a-f]+\",func-name=\"hello_world\(\)\",offset=\"[0-9]+\",size=\"[0-9]+\",inst=\".+?; \\\\\"Hello, World!\\\\\\\\n\\\\\"\"}",
                     "{address=\"0x[0-9a-f]+\",func-name=\"hello_world\(\)\",offset=\"[0-9]+\",size=\"[0-9]+\",inst=\".+?; symbol stub for: printf\"}"])

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_read_memory_bytes_global(self):
        """Test that -data-read-memory-bytes can access global buffers."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Get address of char[] (global)
        self.runCmd("-data-evaluate-expression &g_CharArray")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        addr = int(self.child.after.split("\"")[1], 16)
        size = 5

        # Test that -data-read-memory-bytes works for char[] type (global)
        self.runCmd("-data-read-memory-bytes %#x %d" % (addr, size))
        self.expect(
            "\^done,memory=\[{begin=\"0x0*%x\",offset=\"0x0+\",end=\"0x0*%x\",contents=\"1011121300\"}\]" %
            (addr, addr + size))

        # Get address of static char[]
        self.runCmd("-data-evaluate-expression &s_CharArray")
        self.expect("\^done,value=\"0x[0-9a-f]+\"")
        addr = int(self.child.after.split("\"")[1], 16)
        size = 5

        # Test that -data-read-memory-bytes works for static char[] type
        self.runCmd("-data-read-memory-bytes %#x %d" % (addr, size))
        self.expect(
            "\^done,memory=\[{begin=\"0x0*%x\",offset=\"0x0+\",end=\"0x0*%x\",contents=\"2021222300\"}\]" %
            (addr, addr + size))

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_read_memory_bytes_local(self):
        """Test that -data-read-memory-bytes can access local buffers."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd('-file-exec-and-symbols %s' % self.myexe)
        self.expect(r'\^done')

        # Run to BP_local_array_test_inner
        line = line_number('main.cpp', '// BP_local_array_test_inner')
        self.runCmd('-break-insert main.cpp:%d' % line)
        self.expect(r'\^done,bkpt=\{number="1"')
        self.runCmd('-exec-run')
        self.expect(r'\^running')
        self.expect(r'\*stopped,reason="breakpoint-hit"')

        # Get address of local char[]
        self.runCmd('-data-evaluate-expression "(void *)&array"')
        self.expect(r'\^done,value="0x[0-9a-f]+"')
        addr = int(self.child.after.split('"')[1], 16)
        size = 4

        # Test that an unquoted hex literal address works
        self.runCmd('-data-read-memory-bytes %#x %d' % (addr, size))
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        # Test that a double-quoted hex literal address works
        self.runCmd('-data-read-memory-bytes "%#x" %d' % (addr, size))
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        # Test that unquoted expressions work
        self.runCmd('-data-read-memory-bytes &array %d' % size)
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        # This doesn't work, and perhaps that makes sense, but it does work on
        # GDB
        self.runCmd('-data-read-memory-bytes array 4')
        self.expect(r'\^error')
        #self.expect(r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' % (addr, addr + size))

        self.runCmd('-data-read-memory-bytes &array[2] 2')
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="0304"\}\]' %
            (addr + 2, addr + size))

        self.runCmd('-data-read-memory-bytes first_element_ptr %d' % size)
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        # Test that double-quoted expressions work
        self.runCmd('-data-read-memory-bytes "&array" %d' % size)
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        self.runCmd('-data-read-memory-bytes "&array[0] + 1" 3')
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="020304"\}\]' %
            (addr + 1, addr + size))

        self.runCmd('-data-read-memory-bytes "first_element_ptr + 1" 3')
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="020304"\}\]' %
            (addr + 1, addr + size))

        # Test the -o (offset) option
        self.runCmd('-data-read-memory-bytes -o 1 &array 3')
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="020304"\}\]' %
            (addr + 1, addr + size))

        # Test the --thread option
        self.runCmd('-data-read-memory-bytes --thread 1 &array 4')
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        # Test the --thread option with an invalid value
        self.runCmd('-data-read-memory-bytes --thread 999 &array 4')
        self.expect(r'\^error')

        # Test the --frame option (current frame)
        self.runCmd('-data-read-memory-bytes --frame 0 &array 4')
        self.expect(
            r'\^done,memory=\[\{begin="0x0*%x",offset="0x0+",end="0x0*%x",contents="01020304"\}\]' %
            (addr, addr + size))

        # Test the --frame option (outer frame)
        self.runCmd('-data-read-memory-bytes --frame 1 &array 4')
        self.expect(
            r'\^done,memory=\[\{begin="0x[0-9a-f]+",offset="0x0+",end="0x[0-9a-f]+",contents="05060708"\}\]')

        # Test the --frame option with an invalid value
        self.runCmd('-data-read-memory-bytes --frame 999 &array 4')
        self.expect(r'\^error')

        # Test all the options at once
        self.runCmd(
            '-data-read-memory-bytes --thread 1 --frame 1 -o 2 &array 2')
        self.expect(
            r'\^done,memory=\[\{begin="0x[0-9a-f]+",offset="0x0+",end="0x[0-9a-f]+",contents="0708"\}\]')

        # Test that an expression that references undeclared variables doesn't
        # work
        self.runCmd(
            '-data-read-memory-bytes "&undeclared_array1 + undeclared_array2[1]" 2')
        self.expect(r'\^error')

        # Test that the address argument is required
        self.runCmd('-data-read-memory-bytes')
        self.expect(r'\^error')

        # Test that the count argument is required
        self.runCmd('-data-read-memory-bytes &array')
        self.expect(r'\^error')

        # Test that the address and count arguments are required when other
        # options are present
        self.runCmd('-data-read-memory-bytes --thread 1')
        self.expect(r'\^error')

        self.runCmd('-data-read-memory-bytes --thread 1 --frame 0')
        self.expect(r'\^error')

        # Test that the count argument is required when other options are
        # present
        self.runCmd('-data-read-memory-bytes --thread 1 &array')
        self.expect(r'\^error')

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_list_register_names(self):
        """Test that 'lldb-mi --interpreter' works for -data-list-register-names."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -data-list-register-names: try to get all registers
        self.runCmd("-data-list-register-names")
        self.expect("\^done,register-names=\[\".+?\",")

        # Test -data-list-register-names: try to get specified registers
        self.runCmd("-data-list-register-names 0")
        self.expect("\^done,register-names=\[\".+?\"\]")

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_list_register_values(self):
        """Test that 'lldb-mi --interpreter' works for -data-list-register-values."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test -data-list-register-values: try to get all registers
        self.runCmd("-data-list-register-values x")
        self.expect(
            "\^done,register-values=\[{number=\"0\",value=\"0x[0-9a-f]+\"")

        # Test -data-list-register-values: try to get specified registers
        self.runCmd("-data-list-register-values x 0")
        self.expect(
            "\^done,register-values=\[{number=\"0\",value=\"0x[0-9a-f]+\"}\]")

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_info_line(self):
        """Test that 'lldb-mi --interpreter' works for -data-info-line."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Get the address of main and its line
        self.runCmd("-data-evaluate-expression main")
        self.expect(
            "\^done,value=\"0x[0-9a-f]+ \(a.out`main at main.cpp:[0-9]+\)\"")
        addr = int(self.child.after.split("\"")[1].split(" ")[0], 16)
        line = line_number('main.cpp', '// FUNC_main')

        # Test that -data-info-line works for address
        self.runCmd("-data-info-line *%#x" % addr)
        self.expect(
            "\^done,start=\"0x0*%x\",end=\"0x[0-9a-f]+\",file=\".+?main.cpp\",line=\"%d\"" %
            (addr, line))

        # Test that -data-info-line works for file:line
        self.runCmd("-data-info-line main.cpp:%d" % line)
        self.expect(
            "\^done,start=\"0x0*%x\",end=\"0x[0-9a-f]+\",file=\".+?main.cpp\",line=\"%d\"" %
            (addr, line))

        # Test that -data-info-line fails when invalid address is specified
        self.runCmd("-data-info-line *0x0")
        self.expect(
            "\^error,msg=\"Command 'data-info-line'\. Error: The LineEntry is absent or has an unknown format\.\"")

        # Test that -data-info-line fails when file is unknown
        self.runCmd("-data-info-line unknown_file:1")
        self.expect(
            "\^error,msg=\"Command 'data-info-line'\. Error: The LineEntry is absent or has an unknown format\.\"")

        # Test that -data-info-line fails when line has invalid format
        self.runCmd("-data-info-line main.cpp:bad_line")
        self.expect(
            "\^error,msg=\"error: invalid line number string 'bad_line'")
        self.runCmd("-data-info-line main.cpp:0")
        self.expect("\^error,msg=\"error: zero is an invalid line number")

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_data_evaluate_expression(self):
        """Test that 'lldb-mi --interpreter' works for -data-evaluate-expression."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        line = line_number('main.cpp', '// BP_local_2d_array_test')
        self.runCmd('-break-insert main.cpp:%d' % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check 2d array
        self.runCmd("-data-evaluate-expression array2d")
        self.expect(
            "\^done,value=\"\{\[0\] = \{\[0\] = 1, \[1\] = 2, \[2\] = 3\}, \[1\] = \{\[0\] = 4, \[1\] = 5, \[2\] = 6\}\}\"")
