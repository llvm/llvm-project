.. title:: clang-tidy - cert-dcl58-cpp

cert-dcl58-cpp
==============

Modification of the ``std`` or ``posix`` namespace can result in undefined
behavior.
This check warns for such modifications.
The ``std`` (or ``posix``) namespace is allowed to be extended with (class or
function) template specializations that depend on an user-defined type (a type
that is not defined in the standard system headers).

The check detects the following (user provided) declarations in namespace ``std`` or ``posix``:

- Anything that is not a template specialization.
- Explicit specializations of any standard library function template or class template, if it does not have any user-defined type as template argument.
- Explicit specializations of any member function of a standard library class template.
- Explicit specializations of any member function template of a standard library class or class template.
- Explicit or partial specialization of any member class template of a standard library class or class template.

Examples:

.. code-block:: c++

  namespace std {
    int x; // warning: modification of 'std' namespace can result in undefined behavior [cert-dcl58-cpp]
  }

  namespace posix::a { // warning: modification of 'posix' namespace can result in undefined behavior
  }

  template <>
  struct ::std::hash<long> { // warning: modification of 'std' namespace can result in undefined behavior
    unsigned long operator()(const long &K) const {
      return K;
    }
  };

  struct MyData { long data; };

  template <>
  struct ::std::hash<MyData> { // no warning: specialization with user-defined type
    unsigned long operator()(const MyData &K) const {
      return K.data;
    }
  };

  namespace std {
    template <>
    void swap<bool>(bool &a, bool &b); // warning: modification of 'std' namespace can result in undefined behavior

    template <>
    bool less<void>::operator()<MyData &&, MyData &&>(MyData &&, MyData &&) const { // warning: modification of 'std' namespace can result in undefined behavior
      return true;
    }
  }

This check corresponds to the CERT C++ Coding Standard rule
`DCL58-CPP. Do not modify the standard namespaces
<https://www.securecoding.cert.org/confluence/display/cplusplus/DCL58-CPP.+Do+not+modify+the+standard+namespaces>`_.
