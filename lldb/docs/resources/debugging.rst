Debugging
=========

This page details various ways to debug LLDB itself and other LLDB tools. If
you want to know how to use LLDB in general, please refer to
:doc:`/use/tutorial`.

As LLDB is generally split into 2 tools, ``lldb`` and ``lldb-server``
(``debugserver`` on Mac OS), the techniques shown here will not always apply to
both. With some knowledge of them all, you can mix and match as needed.

In this document we refer to the initial ``lldb`` as the "debugger" and the
program being debugged as the "inferior".

Building For Debugging
----------------------

To build LLDB with debugging information add the following to your CMake
configuration:

::

  -DCMAKE_BUILD_TYPE=Debug \
  -DLLDB_EXPORT_ALL_SYMBOLS=ON

Note that the ``lldb`` you will use to do the debugging does not itself need to
have debug information.

Then build as you normally would according to :doc:`/resources/build`.

If you are going to debug in a way that doesn't need debug info (printf, strace,
etc.) we recommend adding ``LLVM_ENABLE_ASSERTIONS=ON`` to Release build
configurations. This will make LLDB fail earlier instead of continuing with
invalid state (assertions are enabled by default for Debug builds).

Debugging ``lldb``
------------------

The simplest scenario is where we want to debug a local execution of ``lldb``
like this one:

::

  ./bin/lldb test_program

LLDB is like any other program, so you can use the same approach.

::

  ./bin/lldb -- ./bin/lldb /tmp/test.o

That's it. At least, that's the minimum. There's nothing special about LLDB
being a debugger that means you can't attach another debugger to it like any
other program.

What can be an issue is that both debuggers have command line interfaces which
makes it very confusing which one is which:

::

  (the debugger)
  (lldb) run
  Process 1741640 launched: '<...>/bin/lldb' (aarch64)
  Process 1741640 stopped and restarted: thread 1 received signal: SIGCHLD

  (the inferior)
  (lldb) target create "/tmp/test.o"
  Current executable set to '/tmp/test.o' (aarch64).

Another issue is that when you resume the inferior, it will not print the
``(lldb)`` prompt because as far as it knows it hasn't changed state. A quick
way around that is to type something that is clearly not a command and hit
enter.

::

  (lldb) Process 1742266 stopped and restarted: thread 1 received signal: SIGCHLD
  Process 1742266 stopped
  * thread #1, name = 'lldb', stop reason = signal SIGSTOP
      frame #0: 0x0000ffffed5bfbf0 libc.so.6`__GI___libc_read at read.c:26:10
  (lldb) c
  Process 1742266 resuming
  notacommand
  error: 'notacommand' is not a valid command.
  (lldb)

You could just remember whether you are in the debugger or the inferior but
it's more for you to remember, and for interrupt based events you simply may not
be able to know.

Here are some better approaches. First, you could use another debugger like GDB
to debug LLDB. Perhaps an IDE like Xcode or Visual Studio Code. Something which
runs LLDB under the hood so you don't have to type in commands to the debugger
yourself.

Or you could change the prompt text for the debugger and/or inferior.

::

  $ ./bin/lldb -o "settings set prompt \"(lldb debugger) \"" -- \
    ./bin/lldb -o "settings set prompt \"(lldb inferior) \"" /tmp/test.o
  <...>
  (lldb) settings set prompt "(lldb debugger) "
  (lldb debugger) run
  <...>
  (lldb) settings set prompt "(lldb inferior) "
  (lldb inferior)

If you want spacial separation you can run the inferior in one terminal then
attach to it in another. Remember that while paused in the debugger, the inferior
will not respond to input so you will have to ``continue`` in the debugger
first.

::

  (in terminal A)
  $ ./bin/lldb /tmp/test.o

  (in terminal B)
  $ ./bin/lldb ./bin/lldb --attach-pid $(pidof lldb)

Placing Breakpoints
*******************

Generally you will want to hit some breakpoint in the inferior ``lldb``. To place
that breakpoint you must first stop the inferior.

If you're debugging from another window this is done with ``process interrupt``.
The inferior will stop, you place the breakpoint and then ``continue``. Go back
to the inferior and input the command that should trigger the breakpoint.

If you are running debugger and inferior in the same window, input ``ctrl+c``
instead of ``process interrupt`` and then folllow the rest of the steps.

If you are doing this with ``lldb-server`` and find your breakpoint is never
hit, check that you are breaking in code that is actually run by
``lldb-server``. There are cases where code only used by ``lldb`` ends up
linked into ``lldb-server``, so the debugger can break there but the breakpoint
will never be hit.

Debugging ``lldb-server``
-------------------------

Note: If you are on MacOS you are likely using ``debugserver`` instead of
``lldb-server``. The spirit of these instructions applies but the specifics will
be different.

We suggest you read :doc:`/use/remote` before attempting to debug ``lldb-server``
as working out exactly what you want to debug requires that you understand its
various modes and behaviour. While you may not be literally debugging on a
remote target, think of your host machine as the "remote" in this scenario.

The ``lldb-server`` options for your situation will depend on what part of it
or mode you are interested in. To work out what those are, recreate the scenario
first without any extra debugging layers. Let's say we want to debug
``lldb-server`` during the following command:

::

  $ ./bin/lldb /tmp/test.o

We can treat ``lldb-server`` as we treated ``lldb`` before, running it under
``lldb``. The equivalent to having ``lldb`` launch the ``lldb-server`` for us is
to start ``lldb-server`` in the ``gdbserver`` mode.

The following commands recreate that, while debugging ``lldb-server``:

::

  $ ./bin/lldb -- ./bin/lldb-server gdbserver :1234 /tmp/test.o
  (lldb) target create "./bin/lldb-server"
  Current executable set to '<...>/bin/lldb-server' (aarch64).
  <...>
  Process 1742485 launched: '<...>/bin/lldb-server' (aarch64)
  Launched '/tmp/test.o' as process 1742586...

  (in another terminal)
  $ ./bin/lldb /tmp/test.o -o "gdb-remote 1234"

Note that the first ``lldb`` is the one debugging ``lldb-server``. The second
``lldb`` is debugging ``/tmp/test.o`` and is only used to trigger the
interesting code path in ``lldb-server``.

This is another case where you may want to layout your terminals in a
predictable way, or change the prompt of one or both copies of ``lldb``.

If you are debugging a scenario where the ``lldb-server`` starts in ``platform``
mode, but you want to debug the ``gdbserver`` mode you'll have to work out what
subprocess it's starting for the ``gdbserver`` part. One way is to look at the
list of runninng processes and take the command line from there.

In theory it should be possible to use LLDB's
``target.process.follow-fork-mode`` or GDB's ``follow-fork-mode`` to
automatically debug the ``gdbserver`` process as it's created. However this
author has not been able to get either to work in this scenario so we suggest
making a more specific command wherever possible instead.

Output From ``lldb-server``
***************************

As ``lldb-server`` often launches subprocesses, output messages may be hidden
if they are emitted from the child processes.

You can tell it to enable logging using the ``--log-channels`` option. For
example ``--log-channels "posix ptrace"``. However that is not passed on to the
child processes.

The same goes for ``printf``. If it's called in a child process you won't see
the output.

In these cases consider either interactive debugging ``lldb-server`` or
working out a more specific command such that it does not have to spawn a
subprocess. For example if you start with ``platform`` mode, work out what
``gdbserver`` mode process it spawns and run that command instead.

Remote Debugging
----------------

If you want to debug part of LLDB running on a remote machine, the principals
are the same but we will have to start debug servers, then attach debuggers to
those servers.

In the example below we're debugging an ``lldb-server`` ``gdbserver`` mode
command running on a remote machine.

For simplicity we'll use the same ``lldb-server`` as the debug server
and the inferior, but it doesn't need to be that way. You can use ``gdbserver``
(as in, GDB's debug server program) or a system installed ``lldb-server`` if you
suspect your local copy is not stable. As is the case in many of these
scenarios.

::

  $ <...>/bin/lldb-server gdbserver 0.0.0.0:54322 -- \
    <...>/bin/lldb-server gdbserver 0.0.0.0:54321 -- /tmp/test.o

Now we have a debug server listening on port 54322 of our remote (``0.0.0.0``
means it's listening for external connections). This is where we will connect
``lldb`` to, to debug the second ``lldb-server``.

To trigger behaviour in the second ``lldb-server``, we will connect a second
``lldb`` to port 54321 of the remote.

This is the final configuration:

::

  Host                                        | Remote
  --------------------------------------------|--------------------
  lldb A debugs lldb-server on port 54322 ->  | lldb-server A
                                              |  (which runs)
  lldb B debugs /tmp/test.o on port 54321 ->  |    lldb-server B
                                              |      (which runs)
                                              |        /tmp/test.o

You would use ``lldb A`` to place a breakpoint in the code you're interested in,
then ``lldb B`` to trigger ``lldb-server B`` to go into that code and hit the
breakpoint. ``lldb-server A`` is only here to let us debug ``lldb-server B``
remotely.

