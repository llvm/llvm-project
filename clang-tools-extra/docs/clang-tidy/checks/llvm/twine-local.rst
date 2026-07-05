.. title:: clang-tidy - llvm-twine-local

llvm-twine-local
================


Looks for local ``Twine`` variables which are prone to use after frees and
should be generally avoided.

.. code-block:: c++

  static Twine Moo = Twine("bark") + "bah";

  // becomes

  static std::string Moo = (Twine("bark") + "bah").str();

The ``Twine`` does not own the memory of its contents, so it is not
recommended to use ``Twine`` created from temporary strings or string literals.

.. code-block:: c++

  static Twine getModuleIdentifier(StringRef moduleName) {
    return moduleName + "_module";
  }
  void foo() {
    Twine result = getModuleIdentifier(std::string{"abc"} + "def");
    // temporary std::string is destroyed here, result is dangling
  }

After applying this fix-it hints, the code will use ``std::string`` instead of
``Twine`` for local variables. However, ``Twine`` has lots of methods that
are incompatible with ``std::string``, so the user may need to adjust the code
manually after applying the fix-it hints.
