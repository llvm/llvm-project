=============
Hardened Mode
=============

.. contents::
   :local:

.. _using-hardened-mode:

Using the hardened mode
=======================

The hardened mode enables a set of security-critical assertions that prevent
undefined behavior caused by violating preconditions of the standard library.
These assertions can be done with relatively little overhead in constant time
and are intended to be used in production.

In addition to the hardened mode, libc++ also provides the debug mode which
contains all the checks from the hardened mode and additionally more expensive
checks that may affect the complexity of algorithms. The debug mode is intended
to be used for testing, not in production.

Vendors can set the default hardened mode by using the ``LIBCXX_HARDENING_MODE``
CMake variable. Setting ``LIBCXX_HARDENING_MODE`` to ``hardened`` enables the
hardened mode, and similarly setting the variable to ``debug`` enables the debug
mode. The default value is ``unchecked`` which doesn't enable the hardened mode.
Users can control whether the hardened mode or the debug mode is enabled
on a per translation unit basis by setting the ``_LIBCPP_ENABLE_HARDENED_MODE``
or ``_LIBCPP_ENABLE_DEBUG_MODE`` macro to ``1``.

The hardened mode requires ``LIBCXX_ENABLE_ASSERTIONS`` to work. If
``LIBCXX_ENABLE_ASSERTIONS`` was not set explicitly, enabling the hardened mode
(or the debug mode) will implicitly enable ``LIBCXX_ENABLE_ASSERTIONS``. If
``LIBCXX_ENABLE_ASSERTIONS`` was explicitly disabled, this will effectively
disable the hardened mode.

Enabling the hardened mode (or the debug mode) has no impact on the ABI.

Iterator bounds checking
------------------------
TODO(hardening)
