"""
Test lldb process launch flags.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *

class ProcessLaunchTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))

    @not_remote_testsuite_ready
    def test_io (self):
        """Test that process launch I/O redirection flags work properly."""
        self.build ()
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])

        in_file = "input-file.txt"
        out_file = "output-test.out"
        err_file = "output-test.err"

        # Make sure the output files do not exist before launching the process
        try:
            os.remove (out_file)
        except OSError:
            pass

        try:
            os.remove (err_file)
        except OSError:
            pass

        launch_command = "process launch -i " + in_file + " -o " + out_file + " -e " + err_file

        if lldb.remote_platform:
            self.runCmd('platform put-file "{local}" "{remote}"'.format(
                local=in_file, remote=in_file))

        self.expect (launch_command,
                     patterns = [ "Process .* launched: .*a.out" ])

        if lldb.remote_platform:
            self.runCmd('platform get-file "{remote}" "{local}"'.format(
                remote=out_file, local=out_file))
            self.runCmd('platform get-file "{remote}" "{local}"'.format(
                remote=err_file, local=err_file))

        success = True
        err_msg = ""

        # Check to see if the 'stdout' file was created
        try:
            out_f = open (out_file)
        except IOError:
            success = False
            err_msg = err_msg + "   ERROR: stdout file was not created.\n"
        else:
            # Check to see if the 'stdout' file contains the right output
            line = out_f.readline ();
            if line != "This should go to stdout.\n":
                success = False
                err_msg = err_msg + "    ERROR: stdout file does not contain correct output.\n"
                out_f.close();

        # Try to delete the 'stdout' file
        try:
            os.remove (out_file)
        except OSError:
            pass

        # Check to see if the 'stderr' file was created
        try:
            err_f = open (err_file)
        except IOError:
            success = False
            err_msg = err_msg + "     ERROR:  stderr file was not created.\n"
        else:
            # Check to see if the 'stderr' file contains the right output
            line = err_f.readline ()
            if line != "This should go to stderr.\n":
                success = False
                err_msg = err_msg + "    ERROR: stderr file does not contain correct output.\n\
"
                err_f.close()

        # Try to delete the 'stderr' file
        try:
            os.remove (err_file)
        except OSError:
            pass

        if not success:
            self.fail (err_msg)

    # rdar://problem/9056462
    # The process launch flag '-w' for setting the current working directory not working?
    @not_remote_testsuite_ready
    @expectedFailureLinux("llvm.org/pr20265")
    def test_set_working_dir (self):
        """Test that '-w dir' sets the working dir when running the inferior."""
        d = {'CXX_SOURCES' : 'print_cwd.cpp'}
        self.build(dictionary=d)
        self.setTearDownCleanup(d)
        exe = os.path.join (os.getcwd(), "a.out")
        self.runCmd("file " + exe)

        mywd = 'my_working_dir'
        out_file_name = "my_working_dir_test.out"
        err_file_name = "my_working_dir_test.err"

        my_working_dir_path = os.path.join(os.getcwd(), mywd)
        out_file_path = os.path.join(my_working_dir_path, out_file_name)
        err_file_path = os.path.join(my_working_dir_path, err_file_name)

        # Make sure the output files do not exist before launching the process
        try:
            os.remove (out_file_path)
            os.remove (err_file_path)
        except OSError:
            pass

        # Check that we get an error when we have a nonexisting path
        launch_command = "process launch -w %s -o %s -e %s" % (my_working_dir_path + 'z',
                                                               out_file_path,
                                                               err_file_path)

        self.expect(launch_command, error=True,
                patterns = ["error:.* No such file or directory: %sz" % my_working_dir_path])

        # Really launch the process
        launch_command = "process launch -w %s -o %s -e %s" % (my_working_dir_path,
                                                               out_file_path,
                                                               err_file_path)

        self.expect(launch_command,
                    patterns = [ "Process .* launched: .*a.out" ])

        success = True
        err_msg = ""

        # Check to see if the 'stdout' file was created
        try:
            out_f = open(out_file_path)
        except IOError:
            success = False
            err_msg = err_msg + "ERROR: stdout file was not created.\n"
        else:
            # Check to see if the 'stdout' file contains the right output
            line = out_f.readline();
            if self.TraceOn():
                print("line:", line)
            if not re.search(mywd, line):
                success = False
                err_msg = err_msg + "The current working directory was not set correctly.\n"
                out_f.close();

        # Try to delete the 'stdout' and 'stderr' files
        try:
            os.remove(out_file_path)
            os.remove(err_file_path)
            pass
        except OSError:
            pass

        if not success:
            self.fail(err_msg)

    def test_environment_with_special_char (self):
        """Test that environment variables containing '*' and '}' are communicated correctly to the lldb-server."""
        d = {'CXX_SOURCES' : 'print_env.cpp'}
        self.build(dictionary=d)
        self.setTearDownCleanup(d)
        exe = os.path.join (os.getcwd(), "a.out")

        evil_var = 'INIT*MIDDLE}TAIL'

        target = self.dbg.CreateTarget(exe)
        process = target.LaunchSimple(None, ['EVIL=' + evil_var], self.get_process_working_directory())
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)

        out = process.GetSTDOUT(len(evil_var))[:len(evil_var)]
        if out != evil_var:
            self.fail('The environment variable was mis-coded: %s\n' % repr(out))

        newline = process.GetSTDOUT(1)[0]
        if newline != '\r' and newline != '\n':
            self.fail('Garbage at end of environment variable')
