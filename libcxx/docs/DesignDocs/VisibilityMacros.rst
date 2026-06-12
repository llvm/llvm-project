========================
Symbol Visibility Macros
========================

.. contents::
   :local:

.. _visibility-macros:

Overview
========

Libc++ uses various "visibility" macros in order to provide a stable ABI in
both the library and the headers. These macros work by changing the
visibility and inlining characteristics of the symbols they are applied to.

The std namespace also has visibility attributes applied to avoid having to
add visibility macros in as many places. Namespace std has default
type_visibility to export RTTI and other type-specific information. Note that
type_visibility is only supported by Clang, so this doesn't replace
type-specific attributes. The only exception are enums, which GCC always gives
default visibility, thus removing the need for any annotations.

Visibility Macros
=================

**_LIBCPP_HIDDEN**
  Mark a symbol as hidden so it will not be exported from shared libraries.

**_LIBCPP_EXPORTED_FROM_ABI**
  Mark a symbol as being part of our ABI. This includes functions that are part
  of the libc++ library, type information and other symbols. On Windows,
  this macro applies `dllimport`/`dllexport` to the symbol, and on other
  platforms it gives the symbol default visibility. This macro should never be
  used on class templates. On classes it should only be used if the vtable
  lives in the built library.

**_LIBCPP_OVERRIDABLE_FUNC_VIS**
  Mark a symbol as being exported by the libc++ library, but allow it to be
  overridden locally. On non-Windows, this is equivalent to `_LIBCPP_FUNC_VIS`.
  This macro is applied to all `operator new` and `operator delete` overloads.

  **Windows Behavior**: Any symbol marked `dllimport` cannot be overridden
  locally, since `dllimport` indicates the symbol should be bound to a separate
  DLL. All `operator new` and `operator delete` overloads are required to be
  locally overridable, and therefore must not be marked `dllimport`. On Windows,
  this macro therefore expands to `__declspec(dllexport)` when building the
  library and has an empty definition otherwise.

**_LIBCPP_HIDE_FROM_ABI**
  Mark a function as not being part of the ABI of any final linked image that
  uses it.

**_LIBCPP_HIDE_FROM_ABI_AFTER_V1**
  Mark a function as being hidden from the ABI (per `_LIBCPP_HIDE_FROM_ABI`)
  when libc++ is built with an ABI version after ABI v1. This macro is used to
  maintain ABI compatibility for symbols that have been historically exported
  by libc++ in v1 of the ABI, but that we don't want to export in the future.

  This macro works as follows. When we build libc++, we either hide the symbol
  from the ABI (if the symbol is not part of the ABI in the version we're
  building), or we leave it included. From user code (i.e. when we're not
  building libc++), the macro always marks symbols as internal so that programs
  built using new libc++ headers stop relying on symbols that are removed from
  the ABI in a future version. Each time we release a new stable version of the
  ABI, we should create a new _LIBCPP_HIDE_FROM_ABI_AFTER_XXX macro, and we can
  use it to start removing symbols from the ABI after that stable version.

Links
=====

* `[cfe-dev] Visibility in libc++ - 1 <http://lists.llvm.org/pipermail/cfe-dev/2013-July/030610.html>`_
* `[cfe-dev] Visibility in libc++ - 2 <http://lists.llvm.org/pipermail/cfe-dev/2013-August/031195.html>`_
* `[libcxx] Visibility fixes for Windows <http://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20130805/085461.html>`_
