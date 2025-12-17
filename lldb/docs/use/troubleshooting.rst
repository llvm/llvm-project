Troubleshooting
===============

File and Line Breakpoints Are Not Getting Hit
---------------------------------------------

First you must make sure that your source files were compiled with debug
information. Typically this means passing -g to the compiler when compiling
your source file.

When setting breakpoints in implementation source files (.c, cpp, cxx, .m, .mm,
etc), LLDB by default will only search for compile units whose filename
matches. If your code does tricky things like using #include to include source
files:

::

   $ cat foo.c
   #include "bar.c"
   #include "baz.c"
   ...

This will cause breakpoints in "bar.c" to be inlined into the compile unit for
"foo.c". If your code does this, or if your build system combines multiple
files in some way such that breakpoints from one implementation file will be
compiled into another implementation file, you will need to tell LLDB to always
search for inlined breakpoint locations by adding the following line to your
~/.lldbinit file:

::

   $ echo "settings set target.inline-breakpoint-strategy always" >> ~/.lldbinit

This tells LLDB to always look in all compile units and search for breakpoint
locations by file and line even if the implementation file doesn't match.
Setting breakpoints in header files always searches all compile units because
inline functions are commonly defined in header files and often cause multiple
breakpoints to have source line information that matches many header file
paths.

If you set a file and line breakpoint using a full path to the source file,
like Xcode does when setting a breakpoint in its GUI on macOS when you click
in the gutter of the source view, this path must match the full paths in the
debug information. If the paths mismatch, possibly due to passing in a resolved
source file path that doesn't match an unresolved path in the debug
information, this can cause breakpoints to not be resolved. Try setting
breakpoints using the file basename only.

If you are using an IDE and you move your project in your file system and build
again, sometimes doing a clean then build will solve the issue.This will fix
the issue if some .o files didn't get rebuilt after the move as the .o files in
the build folder might still contain stale debug information with the old
source locations.

How Do I Check If I Have Debug Symbols?
---------------------------------------

Checking if a module has any compile units (source files) is a good way to
check if there is debug information in a module:

::

   (lldb) file /tmp/a.out
   (lldb) image list
   [  0] 71E5A649-8FEF-3887-9CED-D3EF8FC2FD6E 0x0000000100000000 /tmp/a.out
         /tmp/a.out.dSYM/Contents/Resources/DWARF/a.out
   [  1] 6900F2BA-DB48-3B78-B668-58FC0CF6BCB8 0x00007fff5fc00000 /usr/lib/dyld
   ....
   (lldb) script lldb.target.module['/tmp/a.out'].GetNumCompileUnits()
   1
   (lldb) script lldb.target.module['/usr/lib/dyld'].GetNumCompileUnits()
   0

Above we can see that "/tmp/a.out" does have a compile unit, and
"/usr/lib/dyld" does not.

We can also list the full paths to all compile units for a module using python:

::

   (lldb) script
   Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
   >>> m = lldb.target.module['a.out']
   >>> for i in range(m.GetNumCompileUnits()):
   ...   cu = m.GetCompileUnitAtIndex(i).file.fullpath
   /tmp/main.c
   /tmp/foo.c
   /tmp/bar.c
   >>>

This can help to show the actual full path to the source files. Sometimes IDEs
will set breakpoints by full paths where the path doesn't match the full path
in the debug info and this can cause LLDB to not resolve breakpoints. You can
use the breakpoint list command with the --verbose option to see the full paths
for any source file and line breakpoints that the IDE set using:

::

   (lldb) breakpoint list --verbose

How Do I Find Out Which Features My Copy Of LLDB Has?
-----------------------------------------------------

Some features such as XML parsing are optional and must be enabled when LLDB is
built. To check which features your copy of LLDB has enabled, use the ``version``
command from within LLDB:

::

   (lldb) version -v

.. note::
   This feature was added in LLDB 22. If you are using an earlier version, you
   can use one of the methods below.

If your LLDB has a scripting langauge enabled, you can also use this command to
print the same information:

::

   (lldb) script lldb.debugger.GetBuildConfiguration()

This command will fail if no scripting langauge was enabled. In that case, you
can instead check the shared library dependencies of LLDB.

For example on Linux you can use the following command:

::

   $ readelf -d <path-to-lldb> | grep NEEDED
   0x0000000000000001 (NEEDED)             Shared library: [liblldb.so.22.0git]
   0x0000000000000001 (NEEDED)             Shared library: [libxml2.so.2]
   0x0000000000000001 (NEEDED)             Shared library: [libedit.so.2]
   <...>

The output above shows us that this particular copy of LLDB has XML parsing
(``libxml2``) and editline (``libedit``) enabled.

.. note::

   ``readelf -d`` as used above only shows direct dependencies of the binary.
   Libraries loaded by a library will not be shown. An example of this is Python.
   ``lldb`` requires ``liblldb`` and it is ``liblldb`` that would require ``libpython``.
   The same ``readelf`` command can be used on ``liblldb`` to see if it does
   depend on Python.

   ``ldd`` will show you the full dependency tree of ``lldb`` but **do not**
   use it unless you trust the ``lldb`` binary. As some versions of ``ldd`` may
   execute the binary in the process of inspecting it.

On Windows, use ``dumpbin /dependents <path-to-lldb>``. The same caveat from
Linux applies to Windows. To find dependencies like Python, you need to run
``dumpbin`` on ``liblldb.dll`` too.

On MacOS, use ``otool -l <path-to-lldb>``.

Why Do I See More, Less, Or Different Registers Than I Expected?
----------------------------------------------------------------

The registers you see in LLDB are defined by information either provided by the
debug server you are connected to, or in some cases, guessed by LLDB.

If you are not seeing the registers you expect, the first step is to figure out
which method is being used to read the register information. They are presented
here in the order that LLDB will try them.

Target Definition Script
^^^^^^^^^^^^^^^^^^^^^^^^

These scripts tell LLDB what registers exist on the debug server without having
to ask it. You can check if you are using one by checking the setting:

::

   (lldb) settings show plugin.process.gdb-remote.target-definition-file

In most cases you will not be using such a script.

If you are using one, or want to write one, you can learn about them by reading
the ``*_target_definition.py`` files in the
`Python examples folder <https://github.com/llvm/llvm-project/tree/main/lldb/examples/python>`__.

.. tip::

   We recommend that before attempting to write a target definition script to solve
   your issues, you look into the other methods first.

Target Description Format XML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most commonly used method is the debug server sending LLDB an XML
document in the
`Target Description Format <https://sourceware.org/gdb/current/onlinedocs/gdb.html/Target-Description-Format.html>`__.

LLDB can only process this if XML parsing was enabled when it was built. A
previous section of this document explains how to check this. If XML is not
enabled and the debug server has offered an XML document, we highly recommend
switching to a build of LLDB with XML parsing enabled.

.. note::

   If LLDB was offered an XML document but could not use it, you will see a
   warning in your debug session to alert you to this situation.

If your LLDB has XML support, next check whether the debug server offered this
XML document. Enable the GDB remote packet log, then connect to the debug server
as you normally would.

::

   (lldb) log enable gdb-remote packets

This will produce a lot of output. Scroll back to just after you connected to
the debug server. Look for lines like these:

::

   lldb             < 104> send packet: $qSupported:xmlRegisters=<...>
   lldb             < 260> read packet: $<...>qXfer:features:read+<...>

The ``sent`` packet is LLDB telling the debug server that it can parse XML
definitions. The ``read`` (received) packet is the debug server telling LLDB
that it is allowed to request Target Description XML.

If you do not see either of these, then one or both of LLDB or the debug server
does not support XML. This is not a fatal problem but may result in degraded
register information.

Switching to a debug server with XML support may not be possible. For example if
you have hardware debug tools that cannot be changed. Consult your local experts
for advice on this.

If you did see those packets, scroll further down to confirm that the XML was
read. You should see output that looks like an XML document:

::

   lldb             <  43> send packet: $qXfer:features:read:target.xml<...>
   lldb             <27583> read packet: $l<?xml version="1.0"?>
   <target version="1.0">
   <...>

Even if the XML document was read by LLDB, it, or parts of it may not be valid.
LLDB is permissive when parsing, so you will not get a fatal error. You need to
check the GDB Remote process log for details of any parsing problems:

::

   (lldb) log enable gdb-remote process

.. note::
   There will always be some messages about register information because we log
   details of successful parsing too. If registers are present but presented in
   an unexpected way, check these log messages to see if LLDB has misinterpreted
   the register information.

If those messages do not tell you what is wrong, your last option is to read the
document yourself by copying it out of the ``gdb-remote packets`` output.
Start by checking the basic XML structure of nested elements and begin/end markers
are correct, then compare it to the Target Description specification.

If reading XML is not possible, or it fails to produce any valid register
information, LLDB falls back to the next method.

qRegisterInfo
^^^^^^^^^^^^^

When XML reading fails, LLDB will try an ``lldb-server`` specific packet
called ``qRegisterInfo``. This packet contains all the same register information
as the XML would, except advanced formatting information. That information tells
LLDB how to format registers as complex types like structures.

So if you are using ``lldb`` connected to ``lldb-server`` or ``debugserver``,
it is likely you will not notice the lack of XML. Connected to anything else,
``qRegisterInfo`` will not work and LLDB will fall back to the final method.

To check if LLDB is using ``qRegisterInfo`` you can check the
``gdb-remote packets`` and ``gdb-remote process`` logs.

Fallback Register Layouts
^^^^^^^^^^^^^^^^^^^^^^^^^

The final method is that LLDB will assume that it can use the general register
read packet and make an assumption about the offset of each register in the
response.

This is often done with older debug servers, or ones running with very limited
resources such as those inside of operating system kernels.

This requires that LLDB and the debug server be in sync which is not always
the case if you are connecting LLDB to anything but ``lldb-server`` or
``debugserver``. Work has been done to make LLDB more compatible, but it is
impossible to predict what might be in use in the wild.

Symptoms of a mismatch are missing, overlapping or incorrect register values.
Often this will cause other features like backtracing to fail.

In this case, assuming you cannot change the debug server, you do not have much
choice but to find a debug client that does match up. Often GDB will work better,
it has a longer history of compatability fixes.

If you must use LLDB, you could patch it to match the debug server. The fallback
layouts are stored in ``lldb/source/Plugins/Process/gdb-remote/GDBRemoteRegisterFallback.cpp``.
