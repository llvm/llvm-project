.. title:: clang-tidy - modernize-make-direct

modernize-make-direct
====================

Replaces ``std::make_*`` function calls with direct constructor calls using class template
argument deduction (CTAD).

================================== ====================================
  Before                             After
---------------------------------- ------------------------------------
``std::make_optional<int>(42)``    ``std::optional(42)``
``std::make_unique<Widget>(1)``    ``std::unique_ptr(new Widget(1))``
``std::make_shared<Widget>(2)``    ``std::shared_ptr(new Widget(2))``
``std::make_pair(1, "test")``      ``std::pair(1, "test")``
================================== ====================================

