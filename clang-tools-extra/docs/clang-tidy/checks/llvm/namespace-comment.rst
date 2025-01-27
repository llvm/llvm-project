.. title:: clang-tidy - llvm-namespace-comment

llvm-namespace-comment
======================

`google-readability-namespace-comments` redirects here as an alias for this
check.

Checks that long namespaces have a closing comment.

https://llvm.org/docs/CodingStandards.html#namespace-indentation

https://google.github.io/styleguide/cppguide.html#Namespaces

.. code-block:: c++

  namespace n1 {
  void f();
  }

  // becomes

  namespace n1 {
  void f();
  }  // namespace n1


Options
-------

.. option:: ShortNamespaceLines

   Requires the closing brace of the namespace definition to be followed by a
   closing comment if the body of the namespace has more than
   `ShortNamespaceLines` lines of code. The value is an unsigned integer that
   defaults to `1U`.

.. option:: SpacesBeforeComments

   An unsigned integer specifying the number of spaces before the comment
   closing a namespace definition. Default is `1U`.

.. option:: AllowNoNamespaceComments

   When `true`, the check will allow that namespace comments are omitted
   entirely. The check only fails if a namespace comment is present but does
   not match namespace definition. Default is `false`.
