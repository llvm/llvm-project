.. title:: clang-tidy - cppcoreguidelines-misleading-capture-default-by-value

cppcoreguidelines-misleading-capture-default-by-value
=====================================================

Warns when lambda specify a by-value capture default and capture ``this``.

By-value capture defaults in member functions can be misleading about whether
data members are captured by value or reference. This occurs because specifying
the capture default ``[=]`` actually captures the ``this`` pointer by value,
not the data members themselves. As a result, data members are still indirectly
accessed via the captured ``this`` pointer, which essentially means they are
being accessed by reference. Therefore, even when using ``[=]``, data members
are effectively captured by reference, which might not align with the user's
expectations.

Examples:

.. code-block:: c++

      struct AClass {
        int member;
        void misleadingLogic() {
          int local = 0;
          member = 0;
          auto f = [=]() mutable {
            local += 1;
            member += 1;
          };
          f();
          // Here, local is 0 but member is 1
        }

        void clearLogic() {
          int local = 0;
          member = 0;
          auto f = [this, local]() mutable {
            local += 1;
            member += 1;
          };
          f();
          // Here, local is 0 but member is 1
        }
      };

This check implements
`CppCoreGuideline F.54 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f54-when-writing-a-lambda-that-captures-this-or-any-class-data-member-dont-use--default-capture>`_.
