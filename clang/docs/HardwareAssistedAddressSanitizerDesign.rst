=======================================================
Hardware-assisted AddressSanitizer Design Documentation
=======================================================

This page is a design document for
**hardware-assisted AddressSanitizer** (or **HWASAN**)
a tool similar to :doc:`AddressSanitizer`,
but based on partial hardware assistance.


Introduction
============

:doc:`AddressSanitizer`
tags every 8 bytes of the application memory with a 1 byte tag (using *shadow memory*),
uses *redzones* to find buffer-overflows and
*quarantine* to find use-after-free.
The redzones, the quarantine, and, to a less extent, the shadow, are the
sources of AddressSanitizer's memory overhead.
See the `AddressSanitizer paper`_ for details.

AArch64 has the `Address Tagging`_ (or top-byte-ignore, TBI), a hardware feature that allows
software to use 8 most significant bits of a 64-bit pointer as
a tag. HWASAN uses `Address Tagging`_
to implement a memory safety tool, similar to :doc:`AddressSanitizer`,
but with smaller memory overhead and slightly different (mostly better)
accuracy guarantees.

Algorithm
=========
* Every heap/stack/global memory object is forcibly aligned by `TG` bytes
  (`TG` is e.g. 16 or 64). We call `TG` the **tagging granularity**.
* For every such object a random `TS`-bit tag `T` is chosen (`TS`, or tag size, is e.g. 4 or 8)
* The pointer to the object is tagged with `T`.
* The memory for the object is also tagged with `T` (using a `TG=>1` shadow memory)
* Every load and store is instrumented to read the memory tag and compare it
  with the pointer tag, exception is raised on tag mismatch.

For a more detailed discussion of this approach see https://arxiv.org/pdf/1802.09517.pdf

Short granules
--------------

A short granule is a granule of size between 1 and `TG-1` bytes. The size
of a short granule is stored at the location in shadow memory where the
granule's tag is normally stored, while the granule's actual tag is stored
in the last byte of the granule. This means that in order to verify that a
pointer tag matches a memory tag, HWASAN must check for two possibilities:

* the pointer tag is equal to the memory tag in shadow memory, or
* the shadow memory tag is actually a short granule size, the value being loaded
  is in bounds of the granule and the pointer tag is equal to the last byte of
  the granule.

Pointer tags between 1 to `TG-1` are possible and are as likely as any other
tag. This means that these tags in memory have two interpretations: the full
tag interpretation (where the pointer tag is between 1 and `TG-1` and the
last byte of the granule is ordinary data) and the short tag interpretation
(where the pointer tag is stored in the granule).

When HWASAN detects an error near a memory tag between 1 and `TG-1`, it
will show both the memory tag and the last byte of the granule. Currently,
it is up to the user to disambiguate the two possibilities.

Instrumentation
===============

Memory Accesses
---------------
All memory accesses are prefixed with an inline instruction sequence that
verifies the tags. Currently, the following sequence is used:

.. code-block:: none

  // int foo(int *a) { return *a; }
  // clang -O2 --target=aarch64-linux -fsanitize=hwaddress -fsanitize-recover=hwaddress -c load.c
  foo:
       0:	90000008 	adrp	x8, 0 <__hwasan_shadow>
       4:	f9400108 	ldr	x8, [x8]         // shadow base (to be resolved by the loader)
       8:	d344dc09 	ubfx	x9, x0, #4, #52  // shadow offset
       c:	38696909 	ldrb	w9, [x8, x9]     // load shadow tag
      10:	d378fc08 	lsr	x8, x0, #56      // extract address tag
      14:	6b09011f 	cmp	w8, w9           // compare tags
      18:	54000061 	b.ne	24 <foo+0x24>    // jump to short tag handler on mismatch
      1c:	b9400000 	ldr	w0, [x0]         // original load
      20:	d65f03c0 	ret
      24:	7100413f 	cmp	w9, #0x10        // is this a short tag?
      28:	54000142 	b.cs	50 <foo+0x50>    // if not, trap
      2c:	12000c0a 	and	w10, w0, #0xf    // find the address's position in the short granule
      30:	11000d4a 	add	w10, w10, #0x3   // adjust to the position of the last byte loaded
      34:	6b09015f 	cmp	w10, w9          // check that position is in bounds
      38:	540000c2 	b.cs	50 <foo+0x50>    // if not, trap
      3c:	9240dc09 	and	x9, x0, #0xffffffffffffff
      40:	b2400d29 	orr	x9, x9, #0xf     // compute address of last byte of granule
      44:	39400129 	ldrb	w9, [x9]         // load tag from it
      48:	6b09011f 	cmp	w8, w9           // compare with pointer tag
      4c:	54fffe80 	b.eq	1c <foo+0x1c>    // if so, continue
      50:	d4212440 	brk	#0x922           // otherwise trap
      54:	b9400000 	ldr	w0, [x0]         // tail duplicated original load (to handle recovery)
      58:	d65f03c0 	ret

Alternatively, memory accesses are prefixed with a function call.
On AArch64, a function call is used by default in trapping mode. The code size
and performance overhead of the call is reduced by using a custom calling
convention that preserves most registers and is specialized to the register
containing the address and the type and size of the memory access.

Heap
----

Tagging the heap memory/pointers is done by `malloc`.
This can be based on any malloc that forces all objects to be TG-aligned.
`free` tags the memory with a different tag.

Stack
-----

Stack frames are instrumented by aligning all non-promotable allocas
by `TG` and tagging stack memory in function prologue and epilogue.

Tags for different allocas in one function are **not** generated
independently; doing that in a function with `M` allocas would require
maintaining `M` live stack pointers, significantly increasing register
pressure. Instead we generate a single base tag value in the prologue,
and build the tag for alloca number `M` as `ReTag(BaseTag, M)`, where
ReTag can be as simple as exclusive-or with constant `M`.

Stack instrumentation is expected to be a major source of overhead,
but could be optional.

Globals
-------

TODO: details.

Error reporting
---------------

Errors are generated by the `HLT` instruction and are handled by a signal handler.

Attribute
---------

HWASAN uses its own LLVM IR Attribute `sanitize_hwaddress` and a matching
C function attribute. An alternative would be to re-use ASAN's attribute
`sanitize_address`. The reasons to use a separate attribute are:

  * Users may need to disable ASAN but not HWASAN, or vise versa,
    because the tools have different trade-offs and compatibility issues.
  * LLVM (ideally) does not use flags to decide which pass is being used,
    ASAN or HWASAN are being applied, based on the function attributes.

This does mean that users of HWASAN may need to add the new attribute
to the code that already uses the old attribute.


Comparison with AddressSanitizer
================================

HWASAN:
  * Is less portable than :doc:`AddressSanitizer`
    as it relies on hardware `Address Tagging`_ (AArch64).
    Address Tagging can be emulated with compiler instrumentation,
    but it will require the instrumentation to remove the tags before
    any load or store, which is infeasible in any realistic environment
    that contains non-instrumented code.
  * May have compatibility problems if the target code uses higher
    pointer bits for other purposes.
  * May require changes in the OS kernels (e.g. Linux seems to dislike
    tagged pointers passed from address space:
    https://www.kernel.org/doc/Documentation/arm64/tagged-pointers.txt).
  * **Does not require redzones to detect buffer overflows**,
    but the buffer overflow detection is probabilistic, with roughly
    `1/(2**TS)` chance of missing a bug (6.25% or 0.39% with 4 and 8-bit TS
    respectively).
  * **Does not require quarantine to detect heap-use-after-free,
    or stack-use-after-return**.
    The detection is similarly probabilistic.

The memory overhead of HWASAN is expected to be much smaller
than that of AddressSanitizer:
`1/TG` extra memory for the shadow
and some overhead due to `TG`-aligning all objects.

Supported architectures
=======================
HWASAN relies on `Address Tagging`_ which is only available on AArch64.
For other 64-bit architectures it is possible to remove the address tags
before every load and store by compiler instrumentation, but this variant
will have limited deployability since not all of the code is
typically instrumented.

The HWASAN's approach is not applicable to 32-bit architectures.


Related Work
============
* `SPARC ADI`_ implements a similar tool mostly in hardware.
* `Effective and Efficient Memory Protection Using Dynamic Tainting`_ discusses
  similar approaches ("lock & key").
* `Watchdog`_ discussed a heavier, but still somewhat similar
  "lock & key" approach.
* *TODO: add more "related work" links. Suggestions are welcome.*


.. _Watchdog: https://www.cis.upenn.edu/acg/papers/isca12_watchdog.pdf
.. _Effective and Efficient Memory Protection Using Dynamic Tainting: https://www.cc.gatech.edu/~orso/papers/clause.doudalis.orso.prvulovic.pdf
.. _SPARC ADI: https://lazytyped.blogspot.com/2017/09/getting-started-with-adi.html
.. _AddressSanitizer paper: https://www.usenix.org/system/files/conference/atc12/atc12-final39.pdf
.. _Address Tagging: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.den0024a/ch12s05s01.html

