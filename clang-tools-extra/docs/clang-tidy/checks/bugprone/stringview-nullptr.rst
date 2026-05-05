.. title:: clang-tidy - bugprone-stringview-nullptr

bugprone-stringview-nullptr
===========================
Finds cases where the ``const CharT*`` constructor of
``std::basic_string_view`` is passed a null pointer argument and replaces them
with calls to the default constructor or construction from the empty string
(``""``) as appropriate.

This prevents code from invoking behavior which is unconditionally undefined.
The single-argument ``const CharT*`` constructor does not check for the null
case before dereferencing its input. In C++23, ``std::basic_string_view``
gained a ``basic_string_view(std::nullptr_t) = delete;`` constructor to
catch some of these cases.

.. code-block:: c++

  std::string_view sv = nullptr;

  sv = nullptr;

  bool is_empty = sv == nullptr;
  bool isnt_empty = sv != nullptr;

  accepts_sv(nullptr);

  accepts_sv({{}});  // A

  accepts_sv({nullptr, 0});  // B

becomes...

.. code-block:: c++

  std::string_view sv = {};

  sv = {};

  bool is_empty = sv == "";
  bool isnt_empty = sv != "";

  accepts_sv("");

  accepts_sv({});  // A

  accepts_sv({nullptr, 0});  // B

.. note::

  The source pattern with trailing comment "A" selects the ``(const CharT*)``
  constructor overload and then value-initializes the pointer, causing a null
  dereference. It happens to not include the ``nullptr`` literal, but it is
  still within the scope of this check.

.. note::

  The source pattern with trailing comment "B" selects the
  ``(const CharT*, size_type)`` constructor which is perfectly valid, since the
  length argument is ``0``. It is not changed by this check.
