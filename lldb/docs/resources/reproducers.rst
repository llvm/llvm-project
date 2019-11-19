Reproducers
===========

As unbelievable as it may sound, the debugger has bugs. These bugs might
manifest themselves as errors, missing results or even a crash. Quite often
these bugs don't reproduce in simple, isolated scenarios. The debugger deals
with a lot of moving parts and subtle differences can easily add up.

Reproducers in LLDB improve the experience for both the users encountering bugs
and the developers working on resolving them. The general idea consists of
*capturing* all the information necessary to later *replay* a debug session
while debugging the debugger.

.. contents::
   :local:

Usage
-----

Reproducers are a generic concept in LLDB and are not inherently coupled with
the command line driver. The functionality can be used for anything that uses
the SB API and the driver is just one example. However, because it's probably
the most common way users interact with lldb, that's the workflow described in
this section.

Capture
```````

Until reproducer capture is enabled by default, you need to launch LLDB in
capture mode. For the command line driver, this means passing ``--capture``.
You cannot enable reproducer capture from within LLDB, as this would be too
late to capture initialization of the debugger.

.. code-block:: bash

  > lldb --capture

In capture mode, LLDB will keep track of all the information it needs to replay
the current debug session. Most data is captured lazily to limit the impact on
performance. To create the reproducer, use the ``reproducer generate``
sub-command. It's always possible to check the status of the reproducers with
the ``reproducer status`` sub-command. Note that generating the reproducer
terminates the debug session.

.. code-block:: none

  (lldb) reproducer status
  Reproducer is in capture mode.
  (lldb) reproducer generate
  Reproducer written to '/path/to/reproducer'
  Please have a look at the directory to assess if you're willing to share the contained information.


The resulting reproducer is a directory. It was a conscious decision to not
compress and archive it automatically. The reproducer can contain potentially
sensitive information like object and symbol files, their paths on disk, debug
information, memory excerpts of the inferior process, etc.

Replay
``````

It is strongly recommended to replay the reproducer locally to ensure it
actually reproduces the expected behavior. If the reproducer doesn't behave
correctly locally, it means there's a bug in the reproducer implementation that
should be addressed.

To replay a reproducer, simply pass its path to LLDB through the ``--replay``
flag. It is unnecessary to pass any other command line flags. The flags that
were passed to LLDB during capture are already part of the reproducer.

.. code-block:: bash

 > lldb --replay /path/to/reproducer


During replay LLDB will behave similar to batch mode. The session should be
identical to the recorded debug session. The only expected differences are that
the binary being debugged doesn't actually run during replay. That means that
you won't see any of its side effects, like things being printed to the
terminal. Another expected difference is the behavior of the ``reproducer
generate`` command, which becomes a NOOP during replay.

Augmenting a Bug Report with a Reproducer
`````````````````````````````````````````

A reproducer can significantly improve a bug report, but it in itself is not
sufficient. Always describe the expected and unexpected behavior. Just like the
debugger can have bugs, the reproducer can have bugs too.


Design
------

Coming soon.

Knows Issues
------------

The reproducers are still a work in progress. Here's a non-exhaustive list of
outstanding work, limitations and known issues.

 - The VFS cannot deal with more than one current working directory. Changing
   the current working directory during the debug session will break relative
   paths.
 - Not all SB APIs are properly instrumented. We need customer serialization
   for APIs that take buffers and lengths.
 - We leak memory during replay because the reproducer doesn't capture the end
   of an object's life time. We need to add instrumentation to the destructor
   of SB API objects.
 - The reproducer includes every file opened by LLDB. This is overkill. For
   example we do not need to capture source files for code listings. There's
   currently no way to say that some file shouldn't be included in the
   reproducer.
 - We do not yet automatically generate a reproducer on a crash. The reason is
   that generating the reproducer is too expensive to do in a signal handler.
   We should re-invoke lldb after a crash and do the heavy lifting.
