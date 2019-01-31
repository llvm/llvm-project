Status
======

macOS
-----

LLDB has matured a lot in the last year and can be used for C, C++ and
Objective-C development for x86_64, i386 and ARM debugging. The entire public
API is exposed though a framework on Mac OS X which is used by Xcode, the lldb
command line tool, and can also be used by Python. The entire public API is
exposed through script bridging which allows LLDB to use an embedded Python
script interpreter, as well as having a Python module named "lldb" which can be
used from Python on the command line. This allows debug sessions to be
scripted. It also allows powerful debugging actions to be created and attached
to a variety of debugging workflows.

Linux
-----

LLDB is improving on Linux. While the debugserver has not been ported (to
enable remote debugging) Linux is nearing feature completeness with Darwin to
debug x86_64 programs, and is partially working with i386 programs. ARM
architectures on Linux are untested. For more details, see the Features by OS
section below.

FreeBSD
-------

LLDB on FreeBSD lags behind the Linux implementation but is improving rapidly.
For more details, see the Features by OS section below.

Windows
-------

LLDB on Windows is still under development, but already useful for i386
programs (x86_64 untested) built with DWARF debug information, including
postmortem analysis of minidumps. For more details, see the Features by OS
section below.

Features Matrix
---------------
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Feature                        | FreeBSD    | Linux                   | Mac OS X (i386/x86_64 and ARM/Thumb) | Windows (i386)       |
|                                | (x86_64)   | (x86_64 and PPC64le)    |                                      |                      |
+================================+============+=========================+======================================+======================+
| Backtracing                    | OK         | OK                      | OK                                   | OK                   |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Breakpoints                    | OK         | OK                      | OK                                   | OK                   |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| C++11:                         | OK         | OK                      | OK                                   | Unknown              |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Commandline lldb tool          | OK         | OK                      | OK                                   | OK                   |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Core file debugging            | OK (ELF)   | OK (ELF)                | OK (MachO)                           | OK (Minidump)        |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Debugserver (remote debugging) | Not ported | Not ported              | OK                                   | Not ported           |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Disassembly                    | OK         | OK                      | OK                                   | OK                   |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Expression evaluation          | Unknown    | Works with some bugs    | OK                                   | Works with some bugs |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| JIT debugging                  | Unknown    | Symbolic debugging only | Untested                             | No                   |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
| Objective-C 2.0:               | Unknown    | Not applicable          | OK                                   | Not applicable       |
+--------------------------------+------------+-------------------------+--------------------------------------+----------------------+
