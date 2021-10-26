subl.. title:: clang-tidy - abseil-no-internal-dependencies

abseil-no-internal-dependencies
===============================

Warns if code using Abseil depends on internal details. If something is in a
namespace that includes the word "internal", code is not allowed to depend upon
it because it's an implementation detail. They cannot friend it, include it,
you mention it or refer to it in any way. Doing so violates Abseil's
compatibility guidelines and may result in breakage. See
https://abseil.io/about/compatibility for more information.

The following cases will result in warnings:

.. code-block:: c++

  absl::strings_internal::foo();
  // warning triggered on this line
  class foo {
    friend struct absl::container_internal::faa;
    // warning triggered on this line
  };
  absl::memory_internal::MakeUniqueResult();
  // warning triggered on this line
