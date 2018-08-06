===============
ShadowCallStack
===============

.. contents::
   :local:

Introduction
============

ShadowCallStack is an **experimental** instrumentation pass, currently only
implemented for x86_64 and aarch64, that protects programs against return
address overwrites (e.g. stack buffer overflows.) It works by saving a
function's return address to a separately allocated 'shadow call stack'
in the function prolog and checking the return address on the stack against
the shadow call stack in the function epilog.

Comparison
----------

To optimize for memory consumption and cache locality, the shadow call stack
stores an index followed by an array of return addresses. This is in contrast
to other schemes, like :doc:`SafeStack`, that mirror the entire stack and
trade-off consuming more memory for shorter function prologs and epilogs with
fewer memory accesses. Similarly, `Return Flow Guard`_ consumes more memory with
shorter function prologs and epilogs than ShadowCallStack but suffers from the
same race conditions (see `Security`_). Intel `Control-flow Enforcement Technology`_
(CET) is a proposed hardware extension that would add native support to
use a shadow stack to store/check return addresses at call/return time. It
would not suffer from race conditions at calls and returns and not incur the
overhead of function instrumentation, but it does require operating system
support.

.. _`Return Flow Guard`: https://xlab.tencent.com/en/2016/11/02/return-flow-guard/
.. _`Control-flow Enforcement Technology`: https://software.intel.com/sites/default/files/managed/4d/2a/control-flow-enforcement-technology-preview.pdf

Compatibility
-------------

ShadowCallStack currently only supports x86_64 and aarch64. A runtime is not
currently provided in compiler-rt so one must be provided by the compiled
application.

On aarch64, the instrumentation makes use of the platform register ``x18``.
On some platforms, ``x18`` is reserved, and on others, it is designated as
a scratch register.  This generally means that any code that may run on the
same thread as code compiled with ShadowCallStack must either target one
of the platforms whose ABI reserves ``x18`` (currently Darwin, Fuchsia and
Windows) or be compiled with the flag ``-ffixed-x18``.

Security
========

ShadowCallStack is intended to be a stronger alternative to
``-fstack-protector``. It protects from non-linear overflows and arbitrary
memory writes to the return address slot; however, similarly to
``-fstack-protector`` this protection suffers from race conditions because of
the call-return semantics on x86_64. There is a short race between the call
instruction and the first instruction in the function that reads the return
address where an attacker could overwrite the return address and bypass
ShadowCallStack. Similarly, there is a time-of-check-to-time-of-use race in the
function epilog where an attacker could overwrite the return address after it
has been checked and before it has been returned to. Modifying the call-return
semantics to fix this on x86_64 would incur an unacceptable performance overhead
due to return branch prediction.

The instrumentation makes use of the ``gs`` segment register on x86_64,
or the ``x18`` register on aarch64, to reference the shadow call stack
meaning that references to the shadow call stack do not have to be stored in
memory. This makes it possible to implement a runtime that avoids exposing
the address of the shadow call stack to attackers that can read arbitrary
memory. However, attackers could still try to exploit side channels exposed
by the operating system `[1]`_ `[2]`_ or processor `[3]`_ to discover the
address of the shadow call stack.

.. _`[1]`: https://eyalitkin.wordpress.com/2017/09/01/cartography-lighting-up-the-shadows/
.. _`[2]`: https://www.blackhat.com/docs/eu-16/materials/eu-16-Goktas-Bypassing-Clangs-SafeStack.pdf
.. _`[3]`: https://www.vusec.net/projects/anc/

On x86_64, leaf functions are optimized to store the return address in a
free register and avoid writing to the shadow call stack if a register is
available. Very short leaf functions are uninstrumented if their execution
is judged to be shorter than the race condition window intrinsic to the
instrumentation.

On aarch64, the architecture's call and return instructions (``bl`` and
``ret``) operate on a register rather than the stack, which means that
leaf functions are generally protected from return address overwrites even
without ShadowCallStack. It also means that ShadowCallStack on aarch64 is not
vulnerable to the same types of time-of-check-to-time-of-use races as x86_64.

Usage
=====

To enable ShadowCallStack, just pass the ``-fsanitize=shadow-call-stack``
flag to both compile and link command lines. On aarch64, you also need to pass
``-ffixed-x18`` unless your target already reserves ``x18``.

Low-level API
-------------

``__has_feature(shadow_call_stack)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases one may need to execute different code depending on whether
ShadowCallStack is enabled. The macro ``__has_feature(shadow_call_stack)`` can
be used for this purpose.

.. code-block:: c

    #if defined(__has_feature)
    #  if __has_feature(shadow_call_stack)
    // code that builds only under ShadowCallStack
    #  endif
    #endif

``__attribute__((no_sanitize("shadow-call-stack")))``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``__attribute__((no_sanitize("shadow-call-stack")))`` on a function
declaration to specify that the shadow call stack instrumentation should not be
applied to that function, even if enabled globally.

Example
=======

The following example code:

.. code-block:: c++

    int foo() {
      return bar() + 1;
    }

Generates the following x86_64 assembly when compiled with ``-O2``:

.. code-block:: gas

    push   %rax
    callq  bar
    add    $0x1,%eax
    pop    %rcx
    retq

or the following aarch64 assembly:

.. code-block:: none

    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      bar
    add     w0, w0, #1
    ldp     x29, x30, [sp], #16
    ret


Adding ``-fsanitize=shadow-call-stack`` would output the following x86_64
assembly:

.. code-block:: gas

    mov    (%rsp),%r10
    xor    %r11,%r11
    addq   $0x8,%gs:(%r11)
    mov    %gs:(%r11),%r11
    mov    %r10,%gs:(%r11)
    push   %rax
    callq  bar
    add    $0x1,%eax
    pop    %rcx
    xor    %r11,%r11
    mov    %gs:(%r11),%r10
    mov    %gs:(%r10),%r10
    subq   $0x8,%gs:(%r11)
    cmp    %r10,(%rsp)
    jne    trap
    retq

    trap:
    ud2

or the following aarch64 assembly:

.. code-block:: none

    str     x30, [x18], #8
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      bar
    add     w0, w0, #1
    ldp     x29, x30, [sp], #16
    ldr     x30, [x18, #-8]!
    ret
