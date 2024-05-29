.. title:: clang-tidy - bugprone-suspicious-stringview-data-usage

bugprone-suspicious-stringview-data-usage
=========================================

Identifies suspicious usages of ``std::string_view::data()`` that could lead to
reading out-of-bounds data due to inadequate or incorrect string null
termination.

It warns when the result of ``data()`` is passed to a constructor or function
without also passing the corresponding result of ``size()`` or ``length()``
member function. Such usage can lead to unintended behavior, particularly when
assuming the data pointed to by ``data()`` is null-terminated.

The absence of a ``c_str()`` method in ``std::string_view`` often leads
developers to use ``data()`` as a substitute, especially when interfacing with
C APIs that expect null-terminated strings. However, since ``data()`` does not
guarantee null termination, this can result in unintended behavior if the API
relies on proper null termination for correct string interpretation.

In today's programming landscape, this scenario can occur when implicitly
converting an ``std::string_view`` to an ``std::string``. Since the constructor
in ``std::string`` designed for string-view-like objects is ``explicit``,
attempting to pass an ``std::string_view`` to a function expecting an
``std::string`` will result in a compilation error. As a workaround, developers
may be tempted to utilize the ``.data()`` method to achieve compilation,
introducing potential risks.

For instance:

.. code-block:: c++

  void printString(const std::string& str) {
    std::cout << "String: " << str << std::endl;
  }

  void something(std::string_view sv) {
    printString(sv.data());
  }

In this example, directly passing ``sv`` to the ``printString`` function would
lead to a compilation error due to the explicit nature of the ``std::string``
constructor. Consequently, developers might opt for ``sv.data()`` to resolve the
compilation error, albeit introducing potential hazards as discussed.

.. option:: StringViewTypes

  Option allows users to specify custom string view-like types for analysis. It
  accepts a semicolon-separated list of type names or regular expressions
  matching these types. Default value is:
  `::std::basic_string_view;::llvm::StringRef`.

.. option:: AllowedCallees

  Specifies methods, functions, or classes where the result of ``.data()`` is
  passed to. Allows to exclude such calls from the analysis. Accepts a
  semicolon-separated list of names or regular expressions matching these
  entities. Default value is: empty string.
