.. title:: clang-tidy - cppcoreguidelines-avoid-capture-default-when-capturing-this

cppcoreguidelines-avoid-capture-default-when-capturing-this
===========================================================

Warns when lambda specify a capture default and capture ``this``.

Capture-defaults in member functions can be misleading about
whether data members are captured by value or reference. For example,
specifying the capture default ``[=]`` will still capture data members
by reference.

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
`CppCoreGuideline F.54 <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f54-if-you-capture-this-capture-all-variables-explicitly-no-default-capture>`_.


Options
-------

.. option:: IgnoreCaptureDefaultByReference

  Do not warn when using capture default by reference. In this case, there is no
  confusion as to whether variables are captured by value or reference.
  Defaults to `false`.
