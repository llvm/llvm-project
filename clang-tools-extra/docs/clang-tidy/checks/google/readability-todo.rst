.. title:: clang-tidy - google-readability-todo

google-readability-todo
=======================

Finds TODO comments without a username or bug number.

The relevant style guide section is
https://google.github.io/styleguide/cppguide.html#TODO_Comments.

Corresponding cpplint.py check: `readability/todo`

Options
-------

.. option:: Style

   A string specifying the TODO style for fix-it hints. Accepted values are
   `Hyphen` and `Parentheses`. Default is `Hyphen`.

   * `Hyphen` will format the fix-it as: ``// TODO: username - details``.
   * `Parentheses` will format the fix-it as: ``// TODO(username): details``.
