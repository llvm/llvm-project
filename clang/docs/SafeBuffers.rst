================
C++ Safe Buffers
================

.. contents::
   :local:


Introduction
============

Clang can be used to harden your C++ code against buffer overflows, an otherwise
common security issue with C-based languages.

The solution described in this document is an integrated programming model as
it combines:

- a family of opt-in Clang warnings (``-Wunsafe-buffer-usage``) emitted at
  during compilation to help you update your code to encapsulate and propagate
  the bounds information associated with pointers;
- runtime assertions implemented as part of
  (`libc++ hardening modes <https://libcxx.llvm.org/Hardening.html>`_)
  that eliminate undefined behavior as long as the coding convention
  is followed and the bounds information is therefore available and correct.

The goal of this work is to enable development of bounds-safe C++ code. It is
not a "push-button" solution; depending on your codebase's existing
coding style, significant (even if largely mechanical) changes to your code
may be necessary. However, it allows you to achieve valuable safety guarantees
on security-critical parts of your codebase.

This solution is under active development. It is already useful for its purpose
but more work is being done to improve ergonomics and safety guarantees
and reduce adoption costs.

The solution aligns in spirit with the "Ranges" safety profile
that was `proposed <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3274r0.pdf>`_
by Bjarne Stroustrup for standardization alongside other C++ safety features.


Pre-Requisites
==============

In order to achieve bounds safety, your codebase needs to have access to
well-encapsulated bounds-safe container, view, and iterator types.
If your project uses libc++, standard container and view types such as
``std::vector`` and ``std::span`` can be made bounds-safe by enabling
the "fast" `hardening mode <https://libcxx.llvm.org/Hardening.html>`_
(passing ``-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST``) to your
compiler) or any of the stricter hardening modes.

In order to harden iterators, you'll need to also obtain a libc++ binary
built with ``_LIBCPP_ABI_BOUNDED_ITERATORS`` -- which is a libc++ ABI setting
that needs to be set for your entire target platform if you need to maintain
binary compatibility with the rest of the platform.

A relatively fresh version of C++ is recommended. In particular, the very useful
standard view class ``std::span`` requires C++20.

Other implementations of the C++ standard library may provide different
flags to enable such hardening.

If you're using custom containers and views, they will need to be hardened
this way as well, but you don't necessarily need to do this ahead of time.

This approach can theoretically be applied to plain C codebases,
assuming that safe primitives are developed to encapsulate all buffer accesses,
acting as "hardened custom containers" to replace raw pointers.
However, such approach would be very unergonomic in C, and safety guarantees
will be lower due to lack of good encapsulation technology. A better approach
to bounds safety for non-C++ programs,
`-fbounds-safety <https://clang.llvm.org/docs/BoundsSafety.html>`_,
is currently in development.

Technically, safety guarantees cannot be provided without hardening
the entire technology stack, including all of your dependencies.
However, applying such hardening technology to even a small portion
of your code may be significantly better than nothing.


The Programming Model for C++
=============================

Assuming that hardened container, view, and iterator classes are available,
what remains is to make sure they are used consistently in your code.
Below we define the specific coding convention that needs to be followed
in order to guarantee safety and how the compiler technology
around ``-Wunsafe-buffer-usage`` assists with that.


Buffer operations should never be performed over raw pointers
-------------------------------------------------------------

Every time a memory access is made, a bounds-safe program must guarantee
that the range of accessed memory addresses falls into the boundaries
of the memory allocated for the object that's being accessed.
In order to establish such a guarantee, the information about such valid range
of addresses -- the **bounds information** associated with the accessed address
-- must be formally available every time a memory access is performed.

A raw pointer does not naturally carry any bounds information.
The bounds information for the pointer may be available *somewhere*, but
it is not associated with the pointer in a formal manner, so a memory access
performed through a raw pointer cannot be automatically verified to be
bounds-safe by the compiler.

That said, the Safe Buffers programming model does **not** try to eliminate
**all** pointer usage. Instead it assumes that most pointers point to
individual objects, not buffers, and therefore they typically aren't
associated with buffer overflow risks. For that reason, in order to identify
the code that requires manual intervention, it is desirable to initially shift
the focus away from the pointers themselves, and instead focus on their
**usage patterns**.

The compiler warning ``-Wunsafe-buffer-usage`` is built to assist you
with this step of the process. A ``-Wunsafe-buffer-usage`` warning is
emitted whenever one of the following **buffer operations** are performed
on a raw pointer:

- array indexing with ``[]``,
- pointer arithmetic,
- bounds-unsafe standard C functions such as ``std::memcpy()``,
- C++ smart pointer operations such as ``std::unique_ptr<T[N]>::operator[]()``,
  which unfortunately cannot be made fully safe within the rules of
  the C++ standard (as of C++23).

This is sufficient for identifying each raw buffer pointer in the program at
**at least one point** during its lifetime across your software stack.

For example, both of the following functions are flagged by
``-Wunsafe-buffer-usage`` because ``pointer`` gets identified as an unsafe
buffer pointer. Even though the second function does not directly access
the buffer, the pointer arithmetic operation inside it may easily be
the only formal "hint" in the program that the pointer does indeed point
to a buffer of multiple objects::

    int get_last_element(int *pointer, size_t size) {
      return ptr[sz - 1]; // warning: unsafe buffer access
    }

    int *get_last_element_ptr(int *pointer, size_t size) {
      return ptr + (size - 1); // warning: unsafe pointer arithmetic
    }


All buffers need to be encapsulated into safe container and view types
----------------------------------------------------------------------

It immediately follows from the previous requirement that once an unsafe pointer
is identified at any point during its lifetime, it should be immediately wrapped
into a safe container type (if the allocation site is "nearby") or a safe
view type (if the allocation site is "far away"). Not only memory accesses,
but also non-access operations such as pointer arithmetic need to be covered
this way in order to benefit from the respective runtime bounds checks.

If a **container** type (``std::array``, ``std::vector``, ``std::string``)
is used for allocating the buffer, this is the best-case scenario because
the container naturally has access to the correct bounds information for the
buffer, and the runtime bounds checks immediately kick in. Additionally,
the container type may provide automatic lifetime management for the buffer
(which may or may not be desirable).

If a **view** type is used (``std::span``, ``std::string_view``), this typically
means that the bounds information for the "adopted" pointer needs to be passed
to the view's constructor manually. This makes runtime checks immediately
kick in with respect to the provided bounds information, which is an immediate
improvement over the raw pointer. However, this situation is still fundamentally
insufficient for security purposes, because **bounds information provided
this way cannot be guaranteed to be correct**.

For example, the function ``get_last_element()`` we've seen in the previous
section can be made **slightly** safer this way::

    int get_last_element(int *pointer, size_t size) {
      std::span<int> sp(pointer, size);
      return sp[size - 1]; // warning addressed
    }

Here ``std::span`` eliminates the potential concern that the operation
``size - 1`` may overflow when ``sz`` is equal to ``0``, leading to a buffer
"underrun". However, such program does not provide a guarantee that
the variable ``sz`` correctly represents the **actual** size fo the buffer
pointed to by ``ptr``. The ``std::span`` constructed this way may be ill-formed.
It may fail to protect you from overrunning the original buffer.

The following example demonstrates one of the most dangerous anti-patterns
of this nature::

    void convert_data(int *source_buf, size_t source_size,
                      int *target_buf, size_t target_size) {
      // Terrible: mismatched pointer / size.
      std::span<int> target_span(target_buf, source_size);
      // ...
    }

The second parameter of ``std::span`` should never be the **desired** size
of the buffer. It should always be the **actual** size of the buffer.
Such code often indicates that the original code has already contained
a vulnerability -- and the use of a safe view class failed to prevent it.

If ``target_span`` actually needs to be of size ``source_size``, a significantly
safer way to produce such a span would be to build it with the correct size
first, and then resize it to the desired size by calling ``.first()``::

    void convert_data(int *source_buf, size_t source_size,
                      int *target_buf, size_t target_size) {
      // Safer.
      std::span<int> target_span(target_buf, target_size).first(source_size);
      // ...
    }

However, these are still half-measures. This code still accepts the
bounds information from the caller in an **informal** manner, and such bounds
information cannot be guaranteed to be correct.

In order to mitigate problems of this nature in their entirety,
the third guideline is imposed.


Encapsulation of bounds information must be respected continuously
------------------------------------------------------------------

The allocation site of the object is the only reliable source of bounds
information for that object. For objects with long lifespans across
multiple functions or even libraries in the software stack, it is essential
to formally preserve the original bounds information as it's being passed
from one piece of code to another.

Standard container and view classes are designed to preserve bounds information
correctly **by construction**. However, they offer a number of ways to "break"
encapsulation, which may cause you to temporarily lose track of the correct
bounds information:

- The two-parameter constructor ``std::span(ptr, size)`` allows you to
  assemble an ill-formed ``std::span``;
- Conversely, you can unwrap a container or a view object into a raw pointer
  and a raw size by calling its ``.data()`` and ``.size()`` methods.
- The overloaded ``operator&()`` found on container and iterator classes
  acts similarly to ``.data()`` in this regard; operations such as
  ``&span[0]`` and ``&*span.begin()`` are effectively unsafe.

Additional ``-Wunsafe-buffer-usage`` warnings are emitted when encapsulation
of **standard** containers is broken in this manner. If you're using
non-standard containers, you can achieve a similar effect with facilities
described in the next section: :ref:`customization`.

For example, our previous attempt to address the warning in
``get_last_element()`` has actually introduced a new warning along the way,
that notifies you about the potentially incorrect bounds information
passed into the two-parameter constructor of ``std::span``::

    int get_last_element(int *pointer, size_t size) {
      std::span<int> sp(pointer, size); // warning: unsafe constructor
      return sp[size - 1];
    }

In order to address this warning, you need to make the function receive
the bounds information from the allocation site in a formal manner.
The function doesn't necessarily need to know where the allocation site is;
it simply needs to be able to accept bounds information **when** it's available.
You can achieve this by refactoring the function to accept a ``std::span``
as a parameter::

    int get_last_element(std::span<int> sp) {
      return sp[size - 1];
    }

This solution puts the responsibility for making sure the span is well-formed
on the **caller**. They should do the same, so that eventually the
responsibility is placed on the allocation site!

Such definition is also very ergonomic as it naturally accepts arbitrary
standard containers without any additional code at the call site::

    void use_last_element() {
      std::vector<int> vec { 1, 2, 3 };
      int x = get_last_element(vec);  // x = 3
    }

Such code is naturally bounds-safe because bounds-information is passed down
from the allocation site to the buffer access site. Only safe operations
are performed on container types. The containers are never "unforged" into
raw pointer-size pairs and never "reforged" again. This is what ideal
bounds-safe C++ code looks like.


.. _customization:

Backwards Compatibility, Interoperation with Unsafe Code, Customization
=======================================================================

Some of the code changes described above can be somewhat intrusive.
For example, changing a function that previously accepted a pointer and a size
separately, to accept a ``std::span`` instead, may require you to update
every call site of the function. This is often undesirable and sometimes
completely unacceptable when backwards compatibility is required.

In order to facilitate **incremental adoption** of the coding convention
described above, as well as to handle various unusual situations, the compiler
provides two additional facilities to give the user more control over
``-Wunsafe-buffer-usage`` diagnostics:

- ``#pragma clang unsafe_buffer_usage`` to mark code as unsafe and **suppress**
  ``-Wunsafe-buffer-usage`` warnings in that code.
- ``[[clang::unsafe_buffer_usage]]`` to annotate potential sources of
  discontinuity of bounds information -- thus introducing
  **additional** ``-Wunsafe-buffer-usage`` warnings.

In this section we describe these facilities in detail and show how they can
help you with various unusual situations.

Suppress unwanted warnings with ``#pragma clang unsafe_buffer_usage``
---------------------------------------------------------------------

If you really need to write unsafe code, you can always suppress all
``-Wunsafe-buffer-usage`` warnings in a section of code by surrounding
that code with the ``unsafe_buffer_usage`` pragma. For example, if you don't
want to address the warning in our example function ``get_last_element()``,
here is how you can suppress it::

    int get_last_element(int *pointer, size_t size) {
      #pragma clang unsafe_buffer_usage begin
      return ptr[sz - 1]; // warning suppressed
      #pragma clang unsafe_buffer_usage end
    }

This behavior is analogous to ``#pragma clang diagnostic`` (`documentation
<https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas>`_)
However, ``#pragma clang unsafe_buffer_usage`` is specialized and recommended
over ``#pragma clang diagnostic`` for a number of technical and non-technical
reasons. Most importantly, ``#pragma clang unsafe_buffer_usage`` is more
suitable for security audits because it is significantly simpler and
describes unsafe code in a more formal manner. On the contrary,
``#pragma clang diagnostic`` comes with a push/pop syntax (as opposed to
the begin/end syntax) and it offers ways to suppress warnings without
mentioning them by name (such as ``-Weverything``), which can make it
difficult to determine at a glance whether the warning is suppressed
on any given line of code.

There are a few natural reasons to use this pragma:

- In implementations of safe custom containers. You need this because ultimately
  ``-Wunsafe-buffer-usage`` cannot help you verify that your custom container
  is safe. It will naturally remind you to audit your container's implementation
  to make sure it has all the necessary runtime checks, but ultimately you'll
  need to suppress it once the audit is complete.
- In performance-critical code where bounds-safety-related runtime checks
  cause an unacceptable performance regression. The compiler can theoretically
  optimize them away (eg. replace a repeated bounds check in a loop with
  a single check before the loop) but it is not guaranteed to do that.
- For incremental adoption purposes. If you want to adopt the coding convention
  gradually, you can always surround an entire file with the
  ``unsafe_buffer_usage`` pragma and then "make holes" in it whenever
  you address warnings on specific portions of the code.
- In the code that interoperates with unsafe code. This may be code that
  will never follow the programming model (such as plain C  code that will
  never be converted to C++) or with the code that simply haven't been converted
  yet.

Interoperation with unsafe code may require a lot of suppressions.
You are encouraged to introduce "unsafe wrapper functions" for various unsafe
operations that you need to perform regularly.

For example, if you regularly receive pointer/size pairs from unsafe code,
you may want to introduce a wrapper function for the unsafe span constructor::

    #pragma clang unsafe_buffer_usage begin

    template <typename T>
    std::span<T> unsafe_forge_span(T *pointer, size_t size) {
      return std::span(pointer, size);
    }

    #pragma clang unsafe_buffer_usage end

Such wrapper function can be used to suppress warnings about unsafe span
constructor usage in a more ergonomic manner::

    void use_unsafe_c_struct(unsafe_c_struct *s) {
      // No warning here.
      std::span<int> sp = unsafe_forge_span(s->pointer, s->size);
      // ...
    }

The code remains unsafe but it also continues to be nicely readable, and it
proves that ``-Wunsafe-buffer-usage`` has done it best to notify you about
the potential unsafety. A security auditor will need to keep an eye on such
unsafe wrappers. **It is still up to you to confirm that the bounds information
passed into the wrapper is correct.**


Flag bounds information discontinuities with ``[[clang::unsafe_buffer_usage]]``
-------------------------------------------------------------------------------

The clang attribute ``[[clang::unsafe_buffer_usage]]``
(`attribute documentation
<https://clang.llvm.org/docs/AttributeReference.html#unsafe-buffer-usage>`_)
allows the user to annotate various objects, such as functions or member
variables, as incompatible with the Safe Buffers programming model.
You are encouraged to do that for arbitrary reasons, but typically the main
reason to do that is when an unsafe function needs to be provided for
backwards compatibility.

For example, in the previous section we've seen how the example function
``get_last_element()`` needed to have its parameter types changed in order
to preserve the continuity of bounds information when receiving a buffer pointer
from the caller. However, such a change breaks both API and ABI compatibility.
The code that previously used this function will no longer compile, nor link,
until every call site of that function is updated. You can reclaim the
backwards compatibility -- in terms of both API and ABI -- by adding
a "compatibility overload"::

    int get_last_element(std::span<int> sp) {
      return sp[size - 1];
    }

    [[clang::unsafe_buffer_usage]] // Please use the new function.
    int get_last_element(int *pointer, size_t size) {
      // Avoid code duplication - simply invoke the safe function!
      // The pragma suppresses the unsafe constructor warning.
      #pragma clang unsafe_buffer_usage begin
      return get_last_element(std::span(pointer, size));
      #pragma clang unsafe_buffer_usage end
    }


Such an overload allows the surrounding code to continue to work.
It is both source-compatible and binary-compatible. It is also strictly safer
than the original function because the unsafe buffer access through raw pointer
is replaced with a safe ``std::span`` access no matter how it's called. However,
because it requires the caller to pass the pointer and the size separately,
it violates our "bounds information continuity" principle. This means that
the callers who care about bounds safety needs to be encouraged to use the
``std::span``-based overload instead. Luckily, the attribute
``[[clang::unsafe_buffer_usage]]`` causes a ``-Wunsafe-buffer-usage`` warning
to be displayed at every call site of the compatibility overload in order to
remind the callers to update their code::

    void use_last_element() {
      std::vector<int> vec { 1, 2, 3 };

      // no warning
      int x = get_last_element(vec);

      // warning: this overload introduces unsafe buffer manipulation
      int x = get_last_element(vec.data(), vec.size());
    }

The compatibility overload can be further simplified with the help of the
``unsafe_forge_span()`` wrapper as described in the previous section --
and it even makes the pragmas unnecessary::

    [[clang::unsafe_buffer_usage]] // Please use the new function.
    int get_last_element(int *pointer, size_t size) {
      // Avoid code duplication - simply invoke the safe function!
      return get_last_element(unsafe_forge_span(pointer, size));
    }

Notice how the attribute ``[[clang::unsafe_buffer_usage]]`` does **not**
suppress the warnings within the function on its own. Similarly, functions whose
entire definitions are covered by ``#pragma clang unsafe_buffer_usage`` do
**not** become automatically annotated with the attribute
``[[clang::unsafe_buffer_usage]]``. They serve two different purposes:

- The pragma says that the function isn't safely **written**;
- The attribute says that the function isn't safe to **use**.

Also notice how we've made an **unsafe** wrapper for a **safe** function.
This is significantly better than making a **safe** wrapper for an **unsafe**
function. In other words, the following solution is significantly more unsafe
and undesirable than the previous solution::

    int get_last_element(std::span<int> sp) {
      // You've just added that attribute, and now you need to
      // immediately suppress the warning that comes with it?
      #pragma clang unsafe_buffer_usage begin
      return get_last_element(sp.data(), sp.size());
      #pragma clang unsafe_buffer_usage end
    }


    [[clang::unsafe_buffer_usage]]
    int get_last_element(int *pointer, size_t size) {
      // This access is still completely unchecked. What's the point of having
      // perfect bounds information if you aren't performing runtime checks?
      #pragma clang unsafe_buffer_usage begin
      return ptr[sz - 1];
      #pragma clang unsafe_buffer_usage end
    }

**Structs and classes**, unlike functions, cannot be overloaded. If a struct
contains an unsafe buffer (in the form of a nested array or a pointer/size pair)
then it is typically impossible to replace them with a safe container (such as
``std::array`` or ``std::span`` respectively) without breaking the layout
of the struct and introducing both source and binary incompatibilities with
the surrounding client code.

Additionally, member variables of a class cannot be naturally "hidden" from
client code. If a class needs to be used by clients who haven't updated to
C++20 yet, you cannot use the C++20-specific ``std::span`` as a member variable
type. If the definition of a struct is shared with plain C code that manipulates
member variables directly, you cannot use any C++-specific types for these
member variables.

In such cases there's usually no backwards-compatible way to use safe types
directly. The best option is usually to discourage the clients from using
member variables directly by annotating the member variables with the attribute
``[[clang::unsafe_buffer_usage]]``, and then to change the interface
of the class to provide safe "accessors" to the unsafe data.

For example, let's assume the worst-case scenario: ``struct foo`` is an unsafe
struct type fully defined in a header shared between plain C code and C++ code::

    struct foo {
      int *pointer;
      size_t size;
    };

In this case you can achieve safety in C++ code by annotating the member
variables as unsafe and encapsulating them into safe accessor methods::

    struct foo {
      [[clang::unsafe_buffer_usage]]
      int *pointer;
      [[clang::unsafe_buffer_usage]]
      size_t size;

    // Avoid showing this code to clients who are unable to digest it.
    #if __cplusplus >= 202002L
      std::span<int> get_pointer_as_span() {
        #pragma clang unsafe_buffer_usage begin
        return std::span(pointer, size);
        #pragma clang unsafe_buffer_usage end
      }

      void set_pointer_from_span(std::span<int> sp) {
        #pragma clang unsafe_buffer_usage begin
        pointer = sp.data();
        size = sp.size();
        #pragma clang unsafe_buffer_usage end
      }

      // Potentially more utility functions.
    #endif
    };

Future Work
===========

The ``-Wunsafe-buffer-usage`` technology is in active development. The warning
is largely ready for everyday use but it is continuously improved to reduce
unnecessary noise as well as cover some of the trickier unsafe operations.

Fix-It Hints for ``-Wunsafe-buffer-usage``
------------------------------------------

A code transformation tool is in development that can semi-automatically
transform large bodies of code to follow the C++ Safe Buffers programming model.
It can currently be accessed by passing the experimental flag
``-fsafe-buffer-usage-suggestions`` in addition to ``-Wunsafe-buffer-usage``.

Fixits produced this way currently assume the default approach described
in this document as they suggest standard containers and views (most notably
``std::span`` and ``std::array``) as replacements for raw buffer pointers.
This also additionally requires libc++ hardening in order to make the runtime
bounds checks actually happen.

Static Analysis to Identify Suspicious Sources of Bounds Information
--------------------------------------------------------------------

The unsafe constructor ``span(pointer, size)`` is often a necessary evil
when it comes to interoperation with unsafe code. However, passing the
correct bounds information to such constructor is often difficult.
In order to detect those ``span(target_pointer, source_size)`` anti-patterns,
path-sensitive analysis performed by `the clang static analyzer
<https://clang-analyzer.llvm.org>`_ can be taught to identify situations
when the pointer and the size are coming from "suspiciously different" sources.

Such analysis will be able to identify the source of information with
significantly higher precision than that of the compiler, making it much better
at identifying incorrect bounds information in your code while producing
significantly fewer warnings. It will also need to bypass
``#pragma clang unsafe_buffer_usage`` suppressions and "see through"
unsafe wrappers such as ``unsafe_forge_span`` -- something that
the static analyzer is naturally capable of doing.
