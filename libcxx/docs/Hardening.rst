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

Vendors can set the default hardening mode by using the
``LIBCXX_HARDENING_MODE`` variable at CMake configuration time. Setting
``LIBCXX_HARDENING_MODE`` to ``hardened`` enables the hardened mode, and
similarly setting the variable to ``debug`` enables the debug mode. The default
value is ``unchecked`` which doesn't enable any hardening.

When hardening is enabled, the compiled library is built with the corresponding
mode enabled, **and** user code will be built with the same mode enabled by
default. If the mode is set to "unchecked" at the CMake configuration time, the
compiled library will not contain any assertions and the default when building
user code will be to have assertions disabled. As a user, you can consult your
vendor to know which level of hardening is enabled by default.

Furthermore, independently of any vendor-selected default, users can always
control which level of hardening is enabled in their code by defining
``_LIBCPP_ENABLE_HARDENED_MODE=0|1`` (or ``_LIBCPP_ENABLE_DEBUG_MODE=0|1``)
before including any libc++ header (we recommend passing
``-D_LIBCPP_ENABLE_HARDENED_MODE=X`` or ``-D_LIBCPP_ENABLE_DEBUG_MODE=X`` to the
compiler). Note that if the compiled library was built by the vendor in the
unchecked mode, functions compiled inside the static or shared library won't
have any hardening enabled even if the user compiles with hardening enabled (the
same is true for the inverse case where the static or shared library was
compiled **with** hardening enabled but the user tries to disable it). However,
most of the code in libc++ is in the headers, so the user-selected value for
``_LIBCPP_ENABLE_HARDENED_MODE`` or ``_LIBCPP_ENABLE_DEBUG_MODE`` (if any) will
usually be respected.

Enabling hardening has no impact on the ABI.

Iterator bounds checking
------------------------
TODO(hardening)
