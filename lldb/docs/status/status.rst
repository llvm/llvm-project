Status
======

FreeBSD
-------

LLDB on FreeBSD lags behind the Linux implementation but is improving rapidly.
For more details, see the Features by OS section below.

Linux
-----

LLDB is improving on Linux. Linux is nearing feature completeness with Darwin
to debug x86_64, i386, ARM, AArch64, IBM POWER (ppc64), IBM Z (s390x), and
MIPS64 programs. For more details, see the Features by OS section below.

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
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Feature               | FreeBSD            | Linux                   | macOS             | NetBSD             | Windows              |
+=======================+====================+=========================+===================+====================+======================+
| Backtracing           | YES                | YES                     | YES               | YES                | YES                  |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Breakpoints           | YES                | YES                     | YES               | YES                | YES                  |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| C++11:                | YES                | YES                     | YES               | YES                | Unknown              |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Commandline tool      | YES                | YES                     | YES               | YES                | YES                  |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Core file debugging   | YES (ELF)          | YES (ELF)               | YES (MachO)       | YES (ELF)          | YES (Minidump)       |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Remote debugging      | YES (lldb-server)  | YES (lldb-server)       | YES (debugserver) | YES (lldb-server)  | NO                   |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Disassembly           | YES                | YES                     | YES               | YES                | YES                  |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Expression evaluation | YES (known issues) | YES (known issues)      | YES               | YES (known issues) | YES (known issues)   |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| JIT debugging         | Unknown            | Symbolic debugging only | Untested          | Work In Progress   | NO                   |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
| Objective-C 2.0:      | Unknown            | N/A                     | YES               | Unknown            | N/A                  |
+-----------------------+--------------------+-------------------------+-------------------+--------------------+----------------------+
