Status
======

FreeBSD
-------

LLDB on FreeBSD lags behind the Linux implementation but is improving rapidly.
For more details, see the Features by OS section below.

Linux
-----

LLDB is improving on Linux. While the debugserver has not been ported (to
enable remote debugging) Linux is nearing feature completeness with Darwin to
debug x86_64 programs, and is partially working with i386 programs. ARM
architectures on Linux are untested. For more details, see the Features by OS
section below.

macOS
-----

LLDB is the system debugger on macOS, iOS, tvOS, and watchOS and
can be used for C, C++, Objective-C and Swift development for x86_64,
i386, ARM, and AArch64 debugging. The entire public API is exposed
through a macOS framework which is used by Xcode and the `lldb`
command line tool. It can also be imported from Python. The entire public API is
exposed through script bridging which allows LLDB to use an embedded Python
script interpreter, as well as having a Python module named "lldb" which can be
used from Python on the command line. This allows debug sessions to be
scripted. It also allows powerful debugging actions to be created and attached
to a variety of debugging workflows.

NetBSD
------

LLDB is improving on NetBSD and reaching feature completeness with Linux.

Windows
-------

LLDB on Windows is still under development, but already useful for i386
programs (x86_64 untested) built with DWARF debug information, including
postmortem analysis of minidumps. For more details, see the Features by OS
section below.

Features Matrix
---------------
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Feature                        | FreeBSD    | Linux                   | macOS      | NetBSD           | Windows              |
+================================+============+=========================+============+==================+======================+
| Backtracing                    | OK         | OK                      | OK         | OK               | OK                   |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Breakpoints                    | OK         | OK                      | OK         | OK               | OK                   |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| C++11:                         | OK         | OK                      | OK         | OK               | Unknown              |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Commandline lldb tool          | OK         | OK                      | OK         | OK               | OK                   |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Core file debugging            | OK (ELF)   | OK (ELF)                | OK (MachO) | OK               | OK (Minidump)        |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Debugserver (remote debugging) | Not ported | OK (lldb-server)        | OK         | OK (lldb-server) | Not ported           |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Disassembly                    | OK         | OK                      | OK         | OK               | OK                   |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Expression evaluation          | Unknown    | Works with some bugs    | OK         | OK (with bugs?)  | Works with some bugs |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| JIT debugging                  | Unknown    | Symbolic debugging only | Untested   | Work In Progress | No                   |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
| Objective-C 2.0:               | Unknown    | Not applicable          | OK         | Unknown          |Not applicable        |
+--------------------------------+------------+-------------------------+------------+------------------+----------------------+
