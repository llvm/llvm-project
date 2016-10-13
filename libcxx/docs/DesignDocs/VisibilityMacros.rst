========================
Symbol Visibility Macros
========================

.. contents::
   :local:

Overview
========

Libc++ uses various "visibility" macros in order to provide a stable ABI in
both the library and the headers. These macros work by changing the
visibility and inlining characteristics of the symbols they are applied to.

Visibility Macros
=================

**_LIBCPP_HIDDEN**
  Mark a symbol as hidden so it will not be exported from shared libraries.

**_LIBCPP_FUNC_VIS**
  Mark a symbol as being exported by the libc++ library. This attribute must
  be applied to the declaration of all functions exported by the libc++ dylib.

**_LIBCPP_INLINE_VISIBILITY**
  Mark a function as hidden and force inlining whenever possible.

**_LIBCPP_ALWAYS_INLINE**
  A synonym for `_LIBCPP_INLINE_VISIBILITY`

**_LIBCPP_TYPE_VIS**
  Mark a type's typeinfo and vtable as having default visibility.
  `_LIBCPP_TYPE_VIS`. This macro has no effect on the visibility of the
  type's member functions. This attribute cannot be used on class templates.

  **GCC Behavior**: GCC does not support Clang's `type_visibility(...)`
  attribute. With GCC the `visibility(...)` attribute is used and member
  functions are affected.

**_LIBCPP_TYPE_VIS_ONLY**
  The same as `_LIBCPP_TYPE_VIS` except that it may be applied to templates.

  **Windows Behavior**: DLLs do not support dllimport/export on class templates.
  The macro has an empty definition on this platform.

  Note: This macro should be renamed `_LIBCPP_TEMPLATE_TYPE_VIS`.

**_LIBCPP_ENUM_VIS**
  Mark the typeinfo of an enum as having default visibility. This attribute
  should be applied to all enum declarations.

  **Windows Behavior**: DLLs do not support importing or exporting enumeration
  typeinfo. The macro has an empty definition on this platform.

  **GCC Behavior**: GCC un-hides the typeinfo for enumerations by default, even
  if `-fvisibility=hidden` is specified. Additionally applying a visibility
  attribute to an enum class results in a warning. The macro has an empty
  definition with GCC.

**_LIBCPP_EXTERN_TEMPLATE_TYPE_VIS**
  Mark the member functions, typeinfo, and vtable of the type named in
  a `_LIBCPP_EXTERN_TEMPLATE` declaration as being exported by the libc++ library.
  This attribute must be specified on all extern class template declarations.

  This macro is used to override the `_LIBCPP_TYPE_VIS_ONLY` attribute
  specified on the primary template and to export the member functions produced
  by the explicit instantiation in the dylib.

  **GCC Behavior**: GCC ignores visibility attributes applied the type in
  extern template declarations and applying an attribute results in a warning.
  However since `_LIBCPP_TYPE_VIS_ONLY` is the same as `_LIBCPP_TYPE_VIS` the
  visibility is already correct. The macro has an empty definition with GCC.

  **Windows Behavior**: `extern template` and `dllexport` are fundamentally
  incompatible *on a template class* on Windows; the former suppresses
  instantiation, while the latter forces it. Specifying both on the same
  declaration makes the template class be instantiated, which is not desirable
  inside headers. This macro therefore expands to `dllimport` outside of libc++
  but nothing inside of it (rather than expanding to `dllexport`); instead, the
  explicit instantiations themselves are marked as exported. Note that this
  applies *only* to extern template *classes*. Extern template *functions* obey
  regular import/export semantics, and applying `dllexport` directly to the
  extern template declaration is the correct thing to do for them.

**_LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS**
  Mark the member functions, typeinfo, and vtable of an explicit instantiation
  of a class template as being exported by the libc++ library. This attribute
  must be specified on all template class explicit instantiations.

  It is only necessary to mark the explicit instantiation itself (as opposed to
  the extern template declaration) as exported on Windows, as discussed above.
  On all other platforms, this macro has an empty definition.

**_LIBCPP_EXTERN_TEMPLATE_INLINE_VISIBILITY**
  Mark a member function of a class template as hidden and inline except when
  building the libc++ library where it marks the symbol as being exported by
  the library.

  This macro is used to maintain ABI compatibility for symbols that have been
  historically exported by the libc++ library but are now marked inline. It
  should only be applied to member functions of class templates that are
  externally instantiated.

**_LIBCPP_EXCEPTION_ABI**
  Mark the member functions, typeinfo, and vtable of the type as being exported
  by the libc++ library. This macro must be applied to all *exception types*.
  Exception types should be defined directly in namespace `std` and not the
  versioning namespace. This allows throwing and catching some exception types
  between libc++ and libstdc++.

**_LIBCPP_NEW_DELETE_VIS**
  Mark a symbol as being exported by the libc++ library. This macro must be
  applied to all `operator new` and `operator delete` overloads.

  **Windows Behavior**: The `operator new` and `operator delete` overloads
  should not be marked as `dllimport`; if they were, source files including the
  `<new>` header (either directly or transitively) would lose the ability to use
  local overloads of `operator new` and `operator delete`. On Windows, this
  macro therefore expands to `__declspec(dllexport)` when building the library
  and has an empty definition otherwise. A related caveat is that libc++ must be
  included on the link line before `msvcrt.lib`, otherwise Microsoft's
  definitions of `operator new` and `operator delete` inside `msvcrt.lib` will
  end up being used instead of libc++'s.

Links
=====

* `[cfe-dev] Visibility in libc++ - 1 <http://lists.llvm.org/pipermail/cfe-dev/2013-July/030610.html>`_
* `[cfe-dev] Visibility in libc++ - 2 <http://lists.llvm.org/pipermail/cfe-dev/2013-August/031195.html>`_
* `[libcxx] Visibility fixes for Windows <http://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20130805/085461.html>`_
