.. _hardening-modes:

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

- **Unchecked mode/none**, which disables all hardening checks.
- **Fast mode**, which contains a set of security-critical checks that can be
  done with relatively little overhead in constant time and are intended to be
  used in production. We recommend most projects adopt this.
- **Extensive mode**, which contains all the checks from fast mode and some
  additional checks for undefined behavior that incur relatively little overhead
  but aren't security-critical. Production builds requiring a broader set of
  checks than fast mode should consider enabling extensive mode. The additional
  rigour impacts performance more than fast mode: we recommend benchmarking to
  determine if that is acceptable for your program.
- **Debug mode**, which enables all the available checks in the library,
  including internal assertions, some of which might be very expensive. This
  mode is intended to be used for testing, not in production.

.. note::

   Enabling hardening has no impact on the ABI.

Notes for users
---------------

As a libc++ user, consult with your vendor to determine the level of hardening
enabled by default.

Users wishing for a different hardening level to their vendor default are able
to control the level by passing **one** of the following options to the compiler:

- ``-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE``
- ``-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST``
- ``-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE``
- ``-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG``

.. warning::

   The exact numeric values of these macros are unspecified and users should not
   rely on them (e.g. expect the values to be sorted in any way).

.. warning::

   If you would prefer to override the hardening level on a per-translation-unit
   basis, you must do so **before** including any headers to avoid `ODR issues`_.

.. _`ODR issues`: https://en.cppreference.com/w/cpp/language/definition#:~:text=is%20ill%2Dformed.-,One%20Definition%20Rule,-Only%20one%20definition

.. note::

   Since the static and shared library components of libc++ are built by the
   vendor, setting this macro will have no impact on the hardening mode for the
   pre-built components. Most libc++ code is header-based, so a user-provided
   value for ``_LIBCPP_HARDENING_MODE`` will be mostly respected.

Notes for vendors
-----------------

Vendors can set the default hardening mode by providing ``LIBCXX_HARDENING_MODE``
as a configuration option, with the possible values of ``none``, ``fast``,
``extensive`` and ``debug``. The default value is ``none`` which doesn't enable
any hardening checks (this mode is sometimes called the ``unchecked`` mode).

This option controls both the hardening mode that the precompiled library is
built with and the default hardening mode that users will build with. If set to
``none``, the precompiled library will not contain any assertions, and user code
will default to building without assertions.

Iterator bounds checking
------------------------

TODO(hardening)
