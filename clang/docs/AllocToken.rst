=================
Allocation Tokens
=================

.. contents::
   :local:

Introduction
============

Clang provides support for allocation tokens to enable allocator-level heap
organization strategies. Clang assigns mode-dependent token IDs to allocation
calls; the runtime behavior depends entirely on the implementation of a
compatible memory allocator.

Possible allocator strategies include:

* **Security Hardening**: Placing allocations into separate, isolated heap
  partitions. For example, separating pointer-containing types from raw data
  can mitigate exploits that rely on overflowing a primitive buffer to corrupt
  object metadata.

* **Memory Layout Optimization**: Grouping related allocations to improve data
  locality and cache utilization.

* **Custom Allocation Policies**: Applying different management strategies to
  different partitions.

Token Assignment Mode
=====================

The default mode to calculate tokens is:

* ``typehashpointersplit``: This mode assigns a token ID based on the hash of
  the allocated type's name, where the top half ID-space is reserved for types
  that contain pointers and the bottom half for types that do not contain
  pointers.

Other token ID assignment modes are supported, but they may be subject to
change or removal. These may (experimentally) be selected with ``-mllvm
-alloc-token-mode=<mode>``:

* ``typehash``: This mode assigns a token ID based on the hash of the allocated
  type's name.

* ``random``: This mode assigns a statically-determined random token ID to each
  allocation site.

* ``increment``: This mode assigns a simple, incrementally increasing token ID
  to each allocation site.

Allocation Token Instrumentation
================================

To enable instrumentation of allocation functions, code can be compiled with
the ``-fsanitize=alloc-token`` flag:

.. code-block:: console

    % clang++ -fsanitize=alloc-token example.cc

The instrumentation transforms allocation calls to include a token ID. For
example:

.. code-block:: c

    // Original:
    ptr = malloc(size);

    // Instrumented:
    ptr = __alloc_token_malloc(size, <token id>);

The following command-line options affect generated token IDs:

* ``-falloc-token-max=<N>``
    Configures the maximum number of tokens. No max by default (tokens bounded
    by ``SIZE_MAX``).

    .. code-block:: console

        % clang++ -fsanitize=alloc-token -falloc-token-max=512 example.cc

Runtime Interface
-----------------

A compatible runtime must be provided that implements the token-enabled
allocation functions. The instrumentation generates calls to functions that
take a final ``size_t token_id`` argument.

.. code-block:: c

    // C standard library functions
    void *__alloc_token_malloc(size_t size, size_t token_id);
    void *__alloc_token_calloc(size_t count, size_t size, size_t token_id);
    void *__alloc_token_realloc(void *ptr, size_t size, size_t token_id);
    // ...

    // C++ operators (mangled names)
    // operator new(size_t, size_t)
    void *__alloc_token__Znwm(size_t size, size_t token_id);
    // operator new[](size_t, size_t)
    void *__alloc_token__Znam(size_t size, size_t token_id);
    // ... other variants like nothrow, etc., are also instrumented.

Fast ABI
--------

An alternative ABI can be enabled with ``-fsanitize-alloc-token-fast-abi``,
which encodes the token ID hint in the allocation function name.

.. code-block:: c

    void *__alloc_token_0_malloc(size_t size);
    void *__alloc_token_1_malloc(size_t size);
    void *__alloc_token_2_malloc(size_t size);
    ...
    void *__alloc_token_0_Znwm(size_t size);
    void *__alloc_token_1_Znwm(size_t size);
    void *__alloc_token_2_Znwm(size_t size);
    ...

This ABI provides a more efficient alternative where
``-falloc-token-max`` is small.

Disabling Instrumentation
-------------------------

To exclude specific functions from instrumentation, you can use the
``no_sanitize("alloc-token")`` attribute:

.. code-block:: c

    __attribute__((no_sanitize("alloc-token")))
    void* custom_allocator(size_t size) {
        return malloc(size);  // Uses original malloc
    }

Note: Independent of any given allocator support, the instrumentation aims to
remain performance neutral. As such, ``no_sanitize("alloc-token")``
functions may be inlined into instrumented functions and vice-versa. If
correctness is affected, such functions should explicitly be marked
``noinline``.

The ``__attribute__((disable_sanitizer_instrumentation))`` is also supported to
disable this and other sanitizer instrumentations.

Suppressions File (Ignorelist)
------------------------------

AllocToken respects the ``src`` and ``fun`` entity types in the
:doc:`SanitizerSpecialCaseList`, which can be used to omit specified source
files or functions from instrumentation.

.. code-block:: bash

    [alloc-token]
    # Exclude specific source files
    src:third_party/allocator.c
    # Exclude function name patterns
    fun:*custom_malloc*
    fun:LowLevel::*

.. code-block:: console

    % clang++ -fsanitize=alloc-token -fsanitize-ignorelist=my_ignorelist.txt example.cc

Conditional Compilation with ``__SANITIZE_ALLOC_TOKEN__``
-----------------------------------------------------------

In some cases, one may need to execute different code depending on whether
AllocToken instrumentation is enabled. The ``__SANITIZE_ALLOC_TOKEN__`` macro
can be used for this purpose.

.. code-block:: c

    #ifdef __SANITIZE_ALLOC_TOKEN__
    // Code specific to -fsanitize=alloc-token builds
    #endif
