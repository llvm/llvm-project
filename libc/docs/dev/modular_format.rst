.. _modular_format:

======================
Modular format strings
======================

Introduction
============

Several C standard library functions (most notably, ``printf`` and ``scanf``),
present a large amount of related features to the caller configured via a
format string. This benefits code size at the caller, since format strings are
typically quite dense, and the equivalent of many individual calls can be
performed with only one. Overall this is a benefit, since to a function calls
typically outnumber the one definition of that function.

However, the implementations of various libc features gated behind aspects of
those format strings can be large enough that they completely swamp the
programs that call them. Floating point and errno conversion in particular can
involve large tables which may be wholly dead. However, due to the format
string structure, this code is dead in a way previously invisible to the
compiler.

To address this, an clang attribute was introduced: ``modular_format(<impl_fn>,
<impl_name>, <aspects>...)``. This adds to the semantics of the existing
``format`` attribute (which must also be present, if implicitly.) The first
argument is a symbol naming a modular version of the implementation; this
version only weakly refers to "aspects" of the implementation that may not be
necessary for certain format strings. The second argument is general
"implementation name" string, and the remaining arguments are a list of handled
aspects of the format string. When the compiler sees that a given call only
needs a fixed set of aspects of the implementation, it may redirect the call to
the implementation function and emit a series of relocations to symbols named
``<impl_name>_<aspect>``. These in turn bring the needed aspects of the call
into the link. The default entrypoints fall the modular ones, except they bring
in every possible implementation aspect.

Mechanism
=========

This functionality is currently gated behind ``LIBC_COPT_PRINTF_MODULAR``. When
set, the ``printf``-family functions gain modular variants, and the regular
variants are modified to call them and emit NONE relocations against all
implementation aspects. 

The implementation aspects are defined in headers using the
``LIBC_PRINTF_MODULE((<decl>), { <body> })`` macro. If
``LIBC_COPT_PRINTF_MODULAR`` is not defined, then this macro makes these
``LIBC_INLINE`` definitions as per usual. Otherwise, for normal usage, these
become weak declarations, which causes any references to the module to become
weak. The implementations are moved to a dedicated impl file for groups of
modules. These define the aspect symbol and the module impls by defining
``LIBC_PRINTF_DEFINE_MODULES`` before including the header. This causes the to
be brought into the link exactly when the aspect symbol is referenced.

Template functions present a special complication: the implementation must
instantiate them for any value that may be used. Since the purpose of the
templates is to implement a fixed interface, the possible arguments should
always be fixed and finite. Accordingly, libc contains def files to enumerate
possible arguments and provide handling for each. Templates are instantiated in
the headers whenever ``LIBC_PRINTF_DEFINE_MODULES`` is defined.

libc and the compiler may understand different sets of aspect names, but their
understanding of what an aspect name means must be identical. libc reports the
set of aspect names that it needs a verdict on, and the compiler will only
provide a verdict for those aspects. If libc asks for a verdict on an aspect
unknown to the compiler, the aspect must be summarily considered to be
required.
