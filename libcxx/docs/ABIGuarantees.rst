.. _ABIGuarantees:

=======================
libc++'s ABI Guarantees
=======================

libc++ provides multiple types of ABI guarantees. These include stability of the layout of structs, the linking of TUs
built against different versions and configurations of the library, and more. This document describes what guarantees
libc++ provides in these different areas as well as what options exist for vendors to affect these guarantees.

Note that all of the guarantees listed below come with an asterisk that there may be circumstances where we deem it
worth it to break that guarantee. These breaks are communicated to vendors by CCing #libcxx-vendors on GitHub. If you
are a vendor, please ask to be added to that group to be notified about changes that potentially affect you.

ABI flags
=========
All the ABI flags listed below can be added to the ``__config_site`` header by the vendor to opt in to an ABI breaking
change. These flags should never be set by the user. When porting libc++ to a new platform, vendord should consider
which flags to enable, assuming that ABI stability is relevant to them. Please contact the libc++ team on Discord or
through other means to be able to make an informed decision on which flags make sense to enable, and to avoid enabling
flags which may not be stable. Flags can be enabled via the ``LIBCXX_ABI_DEFINES`` CMake option.


Stability of the Layout of Structs
==================================

The layout of any user-observable struct is kept stable across versions of the library and any user-facing options
documented :ref:`here <libcxx-configuration-macros>`. There are a lot of structs that have internal names, but are none
the less observable by users; for example through public aliases to these types or because they affect the layout of
other types.

There are multiple ABI flags which affect the layout of certain structs:

``_LIBCPP_ABI_ALTERNATE_STRING_LAYOUT``
---------------------------------------
This changes the internal layout of ``basic_string`` to move the section that is used for the internal buffer to the
front, making it eight byte aligned instead of being unaligned, improving the performance of some operations
significantly.

``_LIBCPP_ABI_NO_ITERATOR_BASES``
---------------------------------
This removes the ``iterator`` base class from ``back_insert_iterator``, ``front_insert_iterator``, ``insert_iterator``,
``istream_iterator``, ``ostream_iterator``, ``ostreambuf_iterator``, ``reverse_iterator``, and ``raw_storage_iterator``.
This doesn't directly affect the layout of these types in most cases, but may result in more padding being used when
they are used in combination, for example ``reverse_iterator<reverse_iterator<T>>``.

``_LIBCPP_ABI_NO_REVERSE_ITERATOR_SECOND_MEMBER``
-------------------------------------------------
This removes a second member in ``reverse_iterator`` that is unused after LWG2360.

``_LIBCPP_ABI_VARIANT_INDEX_TYPE_OPTIMIZATION``
-------------------------------------------------
This changes the index type used inside ``variant`` to the smallest required type to reduce the datasize of variants in
most cases.

``_LIBCPP_ABI_OPTIMIZED_FUNCTION``
----------------------------------
This significantly restructures how ``function`` is written to provide better performance, but is currently not ABI
stable.

``_LIBCPP_ABI_NO_RANDOM_DEVICE_COMPATIBILITY_LAYOUT``
-----------------------------------------------------
This changes the layout of ``random_device`` to only holds state with an implementation that gets entropy from a file
(see ``_LIBCPP_USING_DEV_RANDOM``). When switching from this implementation to another one on a platform that has
already shipped ``random_device``, one needs to retain the same object layout to remain ABI compatible. This flag
removes these workarounds for platforms that don't care about ABI compatibility.

``_LIBCPP_ABI_NO_COMPRESSED_PAIR_PADDING``
------------------------------------------
This removes artificial padding from ``_LIBCPP_COMPRESSED_PAIR`` and ``_LIBCPP_COMPRESSED_TRIPLE``.

These macros are used inside the associative and unordered containers, ``deque``, ``forward_list``, ``future``,
``list``, ``basic_string``, ``function``, ``shared_ptr``, ``unique_ptr``, and ``vector`` to stay ABI compatible with the
legacy ``__compressed_pair`` type. ``__compressed_pair`` had historically been used to reduce storage requirements in
the case of empty types, but has been replaced by ``[[no_unique_address]]``. ``[[no_unique_address]]`` is significantly
lighter in terms of compile time and debug information, and also improves the layout of structs further. However, to
keep ABI stability, the additional improvements in layout had to be reverted by introducing artificial padding. This
flag removes that artificial padding.

``_LIBCPP_ABI_IOS_ALLOW_ARBITRARY_FILL_VALUE``
----------------------------------------------
``basic_ios`` uses ``WEOF`` to indicate that the fill value is uninitialized. However, on platforms where the size of
``char_type`` is equal to or greater than the size of ``int_type`` and ``char_type`` is unsigned,
``char_traits<char_type>::eq_int_type()`` cannot distinguish between ``WEOF`` and ``WCHAR_MAX``. This flag changes
``basic_ios`` to instead track whether the fill value has been initialized using a separate boolean.


Linking TUs which have been compiled against different releases of libc++
=========================================================================
libc++ supports linking TUs which have been compiled against different releases of libc++ by marking symbols with
hidden visibility and changing the mangling of header-only functions in every release.


Linking TUs which have been compiled with different flags affecting code gen
============================================================================
There are a lot of compiler (and library) flags which change the code generated for functions. This includes flags like
``-O1``, which are guaranteed by the compiler to not change the observable behaviour of a correct program, as well as
flags like ``-fexceptions``, which **do** change the observable behaviour. libc++ allows linking of TUs which have been
compiled with specific flags only and makes no guarantees for any of the flags not listed below.

The flags allowed (in any combination) are:
- ``-f[no-]exceptions``
- ``-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE{_FAST,_EXTENSIVE,_DEBUG,_NONE}``

Note that this does not provide any guarantees about user-defined functions, but only that the libc++ functions linked
behave as the flags say.


Availability of symbols in the built library (both static and shared)
=====================================================================
In general, libc++ does not make any guarantees about forwards-compatibility. That is, a TU compiled against new headers
may not work with an older library. Vendors who require such support can leverage availability markup. On the other
hand, backwards compatibility is generally guaranteed.

There are multiple ABI flags that change the symbols exported from the built library:

``_LIBCPP_ABI_DO_NOT_EXPORT_BASIC_STRING_COMMON``
-------------------------------------------------
This removes ``__basic_string_common<true>::__throw_length_error()`` and
``__basic_string_common<true>::__throw_out_of_range()``. These symbols have been used by ``basic_string`` in the past,
but are not referenced from the headers anymore.

``_LIBCPP_ABI_DO_NOT_EXPORT_VECTOR_BASE_COMMON``
------------------------------------------------
This removes ``__vector_base_common<true>::__throw_length_error()`` and
``__vector_base_common<true>::__throw_out_of_range()``. These symbols have been used by ``vector`` in the past, but are
not referenced from the headers anymore.

``_LIBCPP_ABI_DO_NOT_EXPORT_TO_CHARS_BASE_10``
----------------------------------------------
This removes ``__itoa::__u32toa()`` and ``__iota::__u64toa``. These symbols have been used by ``to_chars`` in the past,
but are not referenced from the headers anymore.

``_LIBCPP_ABI_STRING_OPTIMIZED_EXTERNAL_INSTANTIATION``
-------------------------------------------------------
This replaces the symbols that are exported for ``basic_string`` to avoid exporting functions which are likely to be
inlined as well as explicitly moving paths to the built library which are slow, improving fast-path inlining of multiple
functions. This flag is currently unstable.


Stability of the traits of a type
=================================
Whether a particular trait of a type is kept stable depends heavily on the type in question and the trait. The most
important trait of a type to keep stable is the triviality for the purpose of calls, since that directly affects the
function call ABI. Which types are considered non-trivial for the purpose of calls is defined in the
`Itanium ABI <https://itanium-cxx-abi.github.io/cxx-abi/abi.html#definitions>`_.
``is_trivially_copyable`` should also be kept stable usually, since many programs depend on this trait for their own
layouting. This isn't as rigid as the previous requirement though.

There are multiple ABI flags that change traits of a struct:

``_LIBCPP_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI``
---------------------------------------------
This flag adds ``[[clang::trivial_abi]]`` to ``unique_ptr``, which makes it trivial for the purpose of calls.

``_LIBCPP_ABI_ENABLE_SHARED_PTR_TRIVIAL_ABI``
---------------------------------------------
This flag adds ``[[clang::trivial_abi]]`` to ``shared_ptr``, which makes it trivial for the purpose of calls.


Types that public aliases reference
===================================
There are a lot of aliases that reference types with library internal names. For example, containers contain an
``iterator`` alias to a type with a library internal name. These have to always reference the same type, since the
mangling of user-defined function overloads would change otherwise. A notable exception to this are the alias templates
to type traits. There doesn't seem to be anybody who relies on these names staying the same, so it is OK to change what
these aliases actually reference.

There are multiple ABI flags which change which type an alias references:

``_LIBCPP_ABI_INCOMPLETE_TYPES_IN_DEQUE``
-----------------------------------------
This changes ``deque::iterator`` to avoid requiring complete types for ``deque``.

``_LIBCPP_ABI_FIX_UNORDERED_CONTAINER_SIZE_TYPE``
-------------------------------------------------
This changes the unordered container's ``size_types`` aliases.

``_LIBCPP_ABI_USE_WRAP_ITER_IN_STD_ARRAY`` and ``_LIBCPP_ABI_USE_WRAP_ITER_IN_STD_STRING_VIEW``
-----------------------------------------------------------------------------------------------
This changes the ``iterator`` and ``const_iterator`` of ``array`` and ``string_view`` respectively to reference
``__wrap_iter`` instead, which makes it less likely for users to depend on non-portable implementation details. This is
especially useful because enabling bounded iterators hardening requires code not to make these assumptions.

``_LIBCPP_ABI_BOUNDED_ITERATORS``, ``_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STRING``, ``_LIBCPP_ABI_BOUNDED_ITERATORS_IN_VECTOR``, and ``_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY``
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
These flags change the ``iterator`` member of various classes to reference hardened iterators instead. See the
:ref:`hardening documentation <hardening>` for more details.


Meaning of values
=================
The meaning of specific values can usually not be changed, since programs compiled against older versions of the headers
may check for these values. These specific values don't have to be hard-coded, but can also depend on user input.

There are multiple ABI flags that change the meaning of particular values:

``_LIBCPP_ABI_REGEX_CONSTANTS_NONZERO``
---------------------------------------
This changes the value of ``regex_constants::syntax_option-type::ECMAScript`` to be standards-conforming.

``_LIBCPP_ABI_FIX_CITYHASH_IMPLEMENTATION``
-------------------------------------------
This flag fixes the implementation of CityHash used for ``hash<fundamental-type>``. The incorrect implementation of
CityHash has the problem that it drops some bits on the floor. Fixing the implementation changes the hash of values,
resulting in an ABI break.

inline namespaces
=================
Inline namespaces which contain types that are observable by the user need to be kept the same, since they affect
mangling. Almost all of libc++'s symbols are inside an inline namespace. By default that namespace is ``__1``, but can
be changed by the vendor by setting `LIBCXX_ABI_NAMESPACE` during CMake configuration. There is also
``_LIBCPP_ABI_NO_FILESYSTEM_INLINE_NAMESPACE`` to remove the ``__fs`` namespace from surrounding the ``filesystem``
namespace. This shortens the mangling of the filesystem symbols a bit.
