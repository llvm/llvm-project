.. title:: clang-tidy - performance-no-int-to-ptr

performance-no-int-to-ptr
=========================

Diagnoses every integer to pointer cast.

While casting an (integral) pointer to an integer is obvious - you just get
the integral value of the pointer, casting an integer to an (integral) pointer
is deceivingly different. While you will get a pointer with that integral value,
if you got that integral value via a pointer-to-integer cast originally,
the new pointer will lack the provenance information from the original pointer.

So while (integral) pointer to integer casts are effectively no-ops,
and are transparent to the optimizer, integer to (integral) pointer casts
are *NOT* transparent, and may conceal information from optimizer.

While that may be the intention, it is not always so. For example,
let's take a look at a routine to align the pointer up to the multiple of 16:
The obvious, naive implementation for that is:

.. code-block:: c++

  char* src(char* maybe_underbiased_ptr) {
    uintptr_t maybe_underbiased_intptr = (uintptr_t)maybe_underbiased_ptr;
    uintptr_t aligned_biased_intptr = maybe_underbiased_intptr + 15;
    uintptr_t aligned_intptr = aligned_biased_intptr & (~15);
    return (char*)aligned_intptr; // warning: avoid integer to pointer casts [performance-no-int-to-ptr]
  }

The check will rightfully diagnose that cast.

But when provenance concealment is not the goal of the code, but an accident,
this example can be rewritten as follows, without using integer to pointer cast:

.. code-block:: c++

  char*
  tgt(char* maybe_underbiased_ptr) {
      uintptr_t maybe_underbiased_intptr = (uintptr_t)maybe_underbiased_ptr;
      uintptr_t aligned_biased_intptr = maybe_underbiased_intptr + 15;
      uintptr_t aligned_intptr = aligned_biased_intptr & (~15);
      uintptr_t bias = aligned_intptr - maybe_underbiased_intptr;
      return maybe_underbiased_ptr + bias;
  }
