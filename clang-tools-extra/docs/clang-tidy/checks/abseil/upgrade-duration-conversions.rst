.. title:: clang-tidy - abseil-upgrade-duration-conversions

abseil-upgrade-duration-conversions
===================================

Finds calls to ``absl::Duration`` arithmetic operators and factories whose
argument needs an explicit cast to continue compiling after upcoming API
changes.

The operators ``*=``, ``/=``, ``*``, and ``/`` for ``absl::Duration`` currently
accept an argument of class type that is convertible to an arithmetic type. Such
a call currently converts the value to an ``int64_t``, even in a case such as
``std::atomic<float>`` that would result in lossy conversion.

Additionally, the ``absl::Duration`` factory functions (``absl::Hours``,
``absl::Minutes``, etc) currently accept an ``int64_t`` or a floating-point
type. Similar to the arithmetic operators, calls with an argument of class type
that is convertible to an arithmetic type go through the ``int64_t`` path.

These operators and factories will be changed to only accept arithmetic types to
prevent unintended behavior. After these changes are released, passing an
argument of class type will no longer compile, even if the type is implicitly
convertible to an arithmetic type.

Here are example fixes created by this check:

.. code-block:: c++

  std::atomic<int> a;
  absl::Duration d = absl::Milliseconds(a);
  d *= a;

becomes

.. code-block:: c++

  std::atomic<int> a;
  absl::Duration d = absl::Milliseconds(static_cast<int64_t>(a));
  d *= static_cast<int64_t>(a);

Note that this check always adds a cast to ``int64_t`` in order to preserve the
current behavior of user code. It is possible that this uncovers unintended
behavior due to types implicitly convertible to a floating-point type.
