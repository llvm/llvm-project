#!/usr/bin/env python2

import os
import sys
import subprocess
from threading import Timer


hostname = 'localhost'

(r, w) = os.pipe()
args = sys.argv
# Get debugserver, lldb-mi and FileCheck executables' paths with arguments.
debugserver = ' '.join([args[1], '--pipe', str(w), hostname + ':0'])
lldbmi = args[2]
test_file = args[3]
filecheck = 'FileCheck ' + test_file

# Run debugserver, lldb-mi and FileCheck.
debugserver_proc = subprocess.Popen(debugserver.split())
lldbmi_proc = subprocess.Popen(lldbmi, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, shell=True)
filecheck_proc = subprocess.Popen(filecheck, stdin=subprocess.PIPE,
                                  shell=True)

timeout_sec = 30
timer = Timer(timeout_sec, lldbmi_proc.kill)
try:
    timer.start()

    # Get a tcp port chosen by debugserver.
    # The number quite big to get lldb-server's output and to not hang.
    bytes_to_read = 10
    port_bytes = os.read(r, bytes_to_read)
    port = str(port_bytes.decode('utf-8').strip('\x00'))

    with open(test_file, 'r') as f:
        # Replace '$PORT' with a free port number and pass
        # test's content to lldb-mi.
        lldbmi_proc.stdin.write(f.read().replace('$PORT', port))
        out, err = lldbmi_proc.communicate()
        filecheck_proc.stdin.write(out)
finally:
    timer.cancel()

debugserver_proc.kill()
exit(filecheck_proc.returncode)
