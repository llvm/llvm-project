===============
Hardening Modes
===============

.. contents::
   :local:

.. _using-hardening-modes:

Using hardening modes
=====================

libc++ provides several hardening modes, where each mode enables a set of
assertions that prevent undefined behavior caused by violating preconditions of
the standard library. Different hardening modes make different trade-offs
between the amount of checking and runtime performance. The available hardening
modes are:
- fast mode;
- extensive mode;
- debug mode.

The fast mode contains a set of security-critical checks that can be done with
relatively little overhead in constant time and are intended to be used in
production. We recommend most projects to adopt the fast mode.

The extensive mode contains all the checks from the fast mode and additionally
some checks for undefined behavior that incur relatively little overhead but
aren't security-critical. While the performance penalty is somewhat more
significant compared to the fast mode, the extensive mode is still intended to
be usable in production.

The debug mode enables all the available checks in the library, including
internal assertions, some of which might be very expensive. This mode is
intended to be used for testing, not in production.

Vendors can set the default hardening mode by using the
``LIBCXX_HARDENING_MODE`` variable at CMake configuration time with the possible
values of ``none``, ``fast``, ``extensive`` and ``debug``. The default value is
``none`` which doesn't enable any hardening checks (this mode is sometimes
called the ``unchecked`` mode).

When hardening is enabled, the compiled library is built with the corresponding
mode enabled, **and** user code will be built with the same mode enabled by
default. If the mode is set to "none" at the CMake configuration time, the
compiled library will not contain any assertions and the default when building
user code will be to have assertions disabled. As a user, you can consult your
vendor to know which level of hardening is enabled by default.

Furthermore, independently of any vendor-selected default, users can always
control which level of hardening is enabled in their code by defining the macro
``_LIBCPP_HARDENING_MODE`` before including any libc++ headers (preferably by
passing ``-D_LIBCPP_HARDENING_MODE=X`` to the compiler). The macro can be
set to one of the following possible values:

- ``_LIBCPP_HARDENING_MODE_NONE``;
- ``_LIBCPP_HARDENING_MODE_FAST``;
- ``_LIBCPP_HARDENING_MODE_EXTENSIVE``;
- ``_LIBCPP_HARDENING_MODE_DEBUG``.

The exact numeric values of these macros are unspecified and users should not
rely on them (e.g. expect the values to be sorted in any way).

Note that if the compiled library was built by the vendor with the hardening
mode set to "none", functions compiled inside the static or shared library won't
have any hardening enabled even if the user compiles with hardening enabled (the
same is true for the inverse case where the static or shared library was
compiled **with** hardening enabled but the user tries to disable it). However,
most of the code in libc++ is in the headers, so the user-selected value for
``_LIBCPP_HARDENING_MODE``, if any, will usually be respected.

Enabling hardening has no impact on the ABI.

Iterator bounds checking
------------------------
TODO(hardening)
