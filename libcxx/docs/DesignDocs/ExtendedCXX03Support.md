=======================
Extended C++03 Support
=======================

.. contents::
   :local:

Overview
========

libc++ is an implementation of the C++ standard library targeting C++11 or later.

In C++03, the library implements the C++11 standard using C++11 language extensions provided
by Clang.

This document tracks the C++11 extensions libc++ requires, the C++11 extensions it provides,
and how to write minimal C++11 inside libc++.

Required C++11 Compiler Extensions
==================================

Clang provides a large subset of C++11 in C++03 as an extension. The features
libc++ expects Clang  to provide are:

* Variadic templates.
* RValue references and perfect forwarding.
* Alias templates
* defaulted and deleted Functions.
* reference qualified Functions
* ``auto``

There are also features that Clang *does not* provide as an extension in C++03
mode. These include:

* ``constexpr`` and ``noexcept``
*  Trailing return types.
* ``>>`` without a space.


Provided C++11 Library Extensions
=================================

.. warning::
  The C++11 extensions libc++ provides in C++03 are currently undergoing change. Existing extensions
  may be removed in the future. New users are strongly discouraged depending on these extension
  in new code.

  This section will be updated once the libc++ developer community has further discussed the
  future of C++03 with libc++.
