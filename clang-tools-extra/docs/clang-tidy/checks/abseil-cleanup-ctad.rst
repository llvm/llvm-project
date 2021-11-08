.. title:: clang-tidy - abseil-cleanup-ctad

abseil-cleanup-ctad
===================

Suggests switching the initialization pattern of ``absl::Cleanup``
instances from the factory function to class template argument
deduction (CTAD), in C++17 and higher.

.. code-block:: c++

  auto c1 = absl::MakeCleanup([] {});

  const auto c2 = absl::MakeCleanup(std::function<void()>([] {}));

becomes

.. code-block:: c++

  absl::Cleanup c1 = [] {};

  const absl::Cleanup c2 = std::function<void()>([] {});
