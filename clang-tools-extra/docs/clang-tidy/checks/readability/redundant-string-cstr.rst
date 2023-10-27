.. title:: clang-tidy - readability-redundant-string-cstr

readability-redundant-string-cstr
=================================


Finds unnecessary calls to ``std::string::c_str()`` and ``std::string::data()``.

Options
-------

.. option:: StringParameterFunctions

   A semicolon-separated list of (fully qualified) function/method/operator
   names, with the requirement that any parameter currently accepting a
   ``const char*`` input should also be able to accept ``std::string``
   inputs, or proper overload candidates that can do so should exist. This
   can be used to configure functions such as ``fmt::format``,
   ``spdlog::logger::info``, or wrappers around these and similar
   functions. The default value is the empty string.
