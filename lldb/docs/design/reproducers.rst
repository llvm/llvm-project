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


Replay
``````

Reproducers support two replay modes. The main and most common mode is active
replay. It's called active, because it's LLDB that is driving replay by calling
the captured SB API functions one after each other. The second mode is passive
replay. In this mode, LLDB sits idle until an SB API function is called, for
example from Python, and then replays just this individual call.

Active Replay
^^^^^^^^^^^^^

No matter how a reproducer was captured, they can always be replayed with the
command line driver. When a reproducer is passed with the ``--replay`` flag, the
driver short-circuits and passes off control to the reproducer infrastructure,
effectively bypassing its normal operation. This works because the driver is
implemented using the SB API and is therefore nothing more than a sequence of
SB API calls.

Replay is driven by the ``Registry::Replay``. As long as there's data in the
buffer holding the API data, the next SB API function call is deserialized.
Once the function is known, the registry can retrieve its signature, and use
that to deserialize its arguments. The function can then be invoked, most
commonly through the synthesized default replayer, or potentially using a
custom defined replay function. This process continues, until more data is
available or a replay error is encountered.

During replay only a function's side effects matter. The result returned by the
replayed function is ignored because it cannot be observed beyond the driver.
This is sound, because anything that is passed into a subsequent API call will
have been serialized as an input argument. This also works for SB API objects
because the reproducers know about every object that has crossed the API
boundary, which is true by definition for object return values.


Passive Replay
^^^^^^^^^^^^^^

Passive replay exists to support running the API test suite against a
reproducer. The API test suite is written in Python and tests the debugger by
calling into its API from Python. To make this work, the API must transparently
replay itself when called. This is what makes passive replay different from
driver replay, where it is lldb itself that's driving replay. For passive
replay, the driving factor is external.

In order to replay API calls, the reproducers need a way to intercept them.
Every API call is already instrumented with an ``LLDB_RECORD_*`` macro that
captures its input arguments. Furthermore, it also contains the necessary logic
to detect which calls cross the API boundary and should be intercepted. We were
able to reuse all of this to implement passive replay.

During passive replay is enabled, nothing happens until an SB API is called.
Inside that API function, the macro detects whether this call should be
replayed (i.e. crossed the API boundary). If the answer is yes, the next
function is deserialized from the SB API data and compared to the current
function. If the signature matches, we deserialize its input arguments and
reinvoke the current function with the deserialized arguments. We don't need to
do anything special to prevent us from recursively calling the replayed version
again, as the API boundary crossing logic knows that we're still behind the API
boundary when we re-invoked the current function.

Another big difference with driver replay is the return value. While this
didn't matter for driver replay, it's key for passive replay, because that's
what gets checked by the test suite. Luckily, the ``LLDB_RECORD_*`` macros
contained sufficient type information to derive the result type.

Testing
-------

Reproducers are tested in the following ways:

 - Unit tests to cover the reproducer infrastructure. There are tests for the
   provider, loader and for the reproducer instrumentation.
 - Feature specific end-to-end test cases in the ``test/Shell/Reproducer``
   directory. These tests serve as integration and regression tests for the
   reproducers infrastructure, as well as doing some sanity checking for basic
   debugger functionality.
 - The API and shell tests can be run against a replayed reproducer. The
   ``check-lldb-reproducers`` target will run the API and shell test suite
   twice: first running the test normally while capturing a reproducer and then
   a second time using the replayed session as the test input. For the shell
   tests this use a little shim (``lldb-repro``) that uses the arguments and
   current working directory to transparently generate or replay a reproducer.
   For the API tests an extra argument with the reproducer path is passed to
   ``dotest.py`` which initializes the debugger in the appropriate mode.
   Certain tests do not fit this paradigm (for example test that check the
   output of the binary being debugged) and are skipped by marking them as
   unsupported by adding ``UNSUPPORTED: lldb-repro`` to the top of the shell
   test or adding the ``skipIfReproducer`` decorator for the API tests.

Additional testing is possible:

 - It's possible to unconditionally capture reproducers while running the
   entire test suite by setting the ``LLDB_CAPTURE_REPRODUCER`` environment
   variable. Assuming no bugs in reproducers, this can also help to reproduce
   and investigate test failures.

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
