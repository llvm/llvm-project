.. title:: clang-tidy - bugprone-forwarding-reference-overload

bugprone-forwarding-reference-overload
======================================

The check looks for perfect forwarding constructors that can hide copy or move
constructors. If a non const lvalue reference is passed to the constructor, the
forwarding reference parameter will be a better match than the const reference
parameter of the copy constructor, so the perfect forwarding constructor will be
called, which can be confusing.
For detailed description of this issue see: Scott Meyers, Effective Modern C++,
Item 26.

Consider the following example:

.. code-block:: c++

    class Person {
    public:
      // C1: perfect forwarding ctor
      template<typename T>
      explicit Person(T&& n) {}

      // C2: perfect forwarding ctor with parameter default value
      template<typename T>
      explicit Person(T&& n, int x = 1) {}

      // C3: perfect forwarding ctor guarded with enable_if
      template<typename T, typename X = enable_if_t<is_special<T>, void>>
      explicit Person(T&& n) {}

      // C4: variadic perfect forwarding ctor guarded with enable_if
      template<typename... A,
        enable_if_t<is_constructible_v<tuple<string, int>, A&&...>, int> = 0>
      explicit Person(A&&... a) {}

      // (possibly compiler generated) copy ctor
      Person(const Person& rhs);
    };

The check warns for constructors C1 and C2, because those can hide copy and move
constructors. We suppress warnings if the copy and the move constructors are both
disabled (deleted or private), because there is nothing the perfect forwarding
constructor could hide in this case. We also suppress warnings for constructors
like C3 and C4 that are guarded with an ``enable_if``, assuming the programmer was
aware of the possible hiding.

Background
----------

For deciding whether a constructor is guarded with enable_if, we consider the
types of the constructor parameters, the default values of template type parameters
and the types of non-type template parameters with a default literal value. If any
part of these types is ``std::enable_if`` or ``std::enable_if_t``, we assume the
constructor is guarded.

