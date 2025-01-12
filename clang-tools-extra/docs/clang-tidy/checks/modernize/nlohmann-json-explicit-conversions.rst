.. title:: clang-tidy - modernize-nlohmann-json-explicit-conversions

modernize-nlohmann-json-explicit-conversions
============================================

Converts implicit conversions via ``operator ValueType`` in code that uses
the `nlohmann/json`_ library to calls to the ``get()`` member function with
an explicit type. The next major version of the library `will remove
support for`_ these implicit conversions and support for them `can be
disabled now`_ by defining ``JSON_USE_IMPLICIT_CONVERSIONS`` to be ``0``.

.. _nlohmann/json: https://json.nlohmann.me/
.. _will remove support for: https://json.nlohmann.me/integration/migration_guide/#replace-implicit-conversions
.. _can be disabled now: https://json.nlohmann.me/api/macros/json_use_implicit_conversions/

In other words, it turns:

  .. code-block:: c++

     void f(const nlohmann::json &j1, const nlohmann::json &j2)
     {
       int i = j1;
       double d = j2.at("value");
       bool b = *j2.find("valid");
       std::cout << i << " " << d << " " << b << "\n";
     }

into

  .. code-block:: c++

     void f(const nlohmann::json &j1, const nlohmann::json &j2)
     {
       int i = j1.get<int>();
       double d = j2.at("value").get<double>();
       bool b = j2.find("valid")->get<bool>();
       std::cout << i << " " << d << " " << b << "\n";
     }

by knowing what the target type is for the implicit conversion and turning
that into an explicit call to the ``get`` method with that type as the
template parameter.

Unfortunately the check does not work very well if the implicit conversion
occurs in templated code or in a system header. For example, the following
won't be fixed because the implicit conversion will happen inside
``std::optional``'s constructor:

  .. code-block:: c++

     void f(const nlohmann::json &j)
     {
       std::optional<int> oi;
       const auto &it = j.find("value");
       if (it != j.end())
         oi = *it;
       // ...
     }
