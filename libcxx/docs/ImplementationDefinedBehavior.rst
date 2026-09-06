.. _implementation-defined-behavior:

===============================
Implementation-defined behavior
===============================

This document contains the implementation details of the implementation-defined behavior in libc++.
The C++ standard mandates that implementation-defined behavior is documented.

.. note:
   This page is far from complete.


Implementation-defined behavior
===============================

Updating the Time Zone Database
-------------------------------

The C++ standard allows implementations to automatically update the
*remote time zone database*. Libc++ opts not to do that. Instead calling

 - ``std::chrono::remote_version()`` will update the version information of the
   *remote time zone database*,
 - ``std::chrono::reload_tzdb()``, if needed, will update the entire
   *remote time zone database*.

This offers a way for users to update the *remote time zone database* and
give them full control over the process.


`[ostream.formatted.print]/3 <http://eel.is/c++draft/ostream.formatted.print#3>`_ A terminal capable of displaying Unicode
--------------------------------------------------------------------------------------------------------------------------

The C++ standard specifies that the manner in which a stream is determined to refer
to a terminal capable of displaying Unicode is implementation-defined. This is
used for ``std::print`` and similar functions taking an ``ostream&`` argument.

Libc++ determines that a stream is Unicode-capable terminal by:

* First it determines whether the stream's ``rdbuf()`` has an underlying
  ``FILE*``. This is ``true`` in the following cases:

  * The stream is ``std::cout``, ``std::cerr``, or ``std::clog``.

  * A ``std::basic_filebuf<CharT, Traits>`` derived from ``std::filebuf``.

* The way to determine whether this ``FILE*`` refers to a terminal capable of
  displaying Unicode is the same as specified for `void vprint_unicode(FILE*
  stream, string_view fmt, format_args args);
  <http://eel.is/c++draft/print.fun#7>`_. This function is used for other
  ``std::print`` overloads that don't take an ``ostream&`` argument.

`[sf.cmath] <https://wg21.link/sf.cmath>`_ Mathematical Special Functions: Large indices
----------------------------------------------------------------------------------------

Most functions within the Mathematical Special Functions section contain integral indices.
The C++ standard specifies the result for larger indices as implementation-defined.
Libc++ pursuits reasonable results by choosing the same formulas as for indices below that threshold.
E.g.,

- ``std::hermite(unsigned n, T x)`` for ``n >= 128``


`[filebuf.virtuals] <https://eel.is/c++draft/filebuf.virtual>`_ Effect of calling ``basic_filebuf::setbuf`` with nonzero arguments
----------------------------------------------------------------------------------------------------------------------------------

Libc++ uses the provided buffer as the underlying buffer for input and output, and
does not discard that buffer even when the stream is closed.


`[stringbuf.cons] <http://eel.is/c++draft/stringbuf.cons>`_ Whether sequence pointers are initialized to null pointers
----------------------------------------------------------------------------------------------------------------------

Libc++ does not initialize the pointers to null pointers. It resizes the buffer
to its capacity and uses that size. This means the SSO buffer of
``std::string`` is used as initial output buffer.

`[debugging.utility] <https://eel.is/c++draft/debugging#lib:is_debugger_present>`_ is_debugger_present
------------------------------------------------------------------------------------------------------

Libc++ implements `std::is_debugger_present()` differently depending on the host platform:

- Linux: ``/proc/self/status`` is read for the ``TracerPid`` entry, and returns true if it is not equal to 0.
  Do note that this will return true, if an executable is run under a utility that traces a process, but isn't
  necessarily a debugger, e.g. ``strace``.

- Darwin & FreeBSD: ``sysctl`` is called to retrieve the current process information and check if the ``P_TRACED``
  flag is set in the returned info.
  On Darwin, ``kinfo_proc::kp_proc.p_flag`` is checked.
  On FreeBSD, ``kinfo_proc::ki_flag`` is checked.

- AIX: Reads the file ``/proc/{current_pid}/status`` (subtituting current_pid for the process' PID) into a
  ``pstatus_t`` structure, then checks if the ``STRC`` flag is set in ``pstatus_t::pr_flag``.

- Windows: Calls ``IsDebuggerPresent()``.

It is also defined as a weak symbol in the library to allow it to be replaceable, as per the standard, so a user may
redefine it with their own implementation.


Listed in the index of implementation-defined behavior
======================================================

The order of the entries matches the entries in the
`draft of the Standard <http://eel.is/c++draft/impldefindex>`_.
