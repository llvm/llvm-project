.. title:: clang-tidy - bugprone-stringview-nullptr

bugprone-stringview-nullptr
===========================
Checks for various ways that the ``const CharT*`` constructor of
``std::basic_string_view`` can be passed a null argument and replaces them
with the default constructor in most cases. For the comparison operators,
braced initializer list does not compile so instead a call to ``.empty()``
or the empty string literal are used, where appropriate.

This prevents code from invoking behavior which is unconditionally undefined.
The single-argument ``const CharT*`` constructor does not check for the null
case before dereferencing its input. The standard is slated to add an
explicitly-deleted overload to catch some of these cases: wg21.link/p2166

To catch the additional cases of ``NULL`` (which expands to ``__null``) and
``0``, first run the ``modernize-use-nullptr`` check to convert the callers to
``nullptr``.

.. code-block:: c++

  std::string_view sv = nullptr;

  sv = nullptr;

  bool is_empty = sv == nullptr;
  bool isnt_empty = sv != nullptr;

  accepts_sv(nullptr);

  accepts_sv({{}});  // A

  accepts_sv({nullptr, 0});  // B

is translated into...

.. code-block:: c++

  std::string_view sv = {};

  sv = {};

  bool is_empty = sv.empty();
  bool isnt_empty = !sv.empty();

  accepts_sv("");

  accepts_sv("");  // A

  accepts_sv({nullptr, 0});  // B

.. note::

  The source pattern with trailing comment "A" selects the ``(const CharT*)``
  constructor overload and then value-initializes the pointer, causing a null
  dereference. It happens to not include the ``nullptr`` literal, but it is
  still within the scope of this ClangTidy check.

.. note::

  The source pattern with trailing comment "B" selects the
  ``(const CharT*, size_type)`` constructor which is perfectly valid, since the
  length argument is ``0``. It is not changed by this ClangTidy check.
