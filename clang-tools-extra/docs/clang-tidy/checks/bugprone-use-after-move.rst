.. title:: clang-tidy - bugprone-use-after-move

bugprone-use-after-move
=======================

Warns if an object is used after it has been moved, for example:

  .. code-block:: c++

    std::string str = "Hello, world!\n";
    std::vector<std::string> messages;
    messages.emplace_back(std::move(str));
    std::cout << str;

The last line will trigger a warning that ``str`` is used after it has been
moved.

The check does not trigger a warning if the object is reinitialized after the
move and before the use. For example, no warning will be output for this code:

  .. code-block:: c++

    messages.emplace_back(std::move(str));
    str = "Greetings, stranger!\n";
    std::cout << str;

The check takes control flow into account. A warning is only emitted if the use
can be reached from the move. This means that the following code does not
produce a warning:

  .. code-block:: c++

    if (condition) {
      messages.emplace_back(std::move(str));
    } else {
      std::cout << str;
    }

On the other hand, the following code does produce a warning:

  .. code-block:: c++

    for (int i = 0; i < 10; ++i) {
      std::cout << str;
      messages.emplace_back(std::move(str));
    }

(The use-after-move happens on the second iteration of the loop.)

In some cases, the check may not be able to detect that two branches are
mutually exclusive. For example (assuming that ``i`` is an int):

  .. code-block:: c++

    if (i == 1) {
      messages.emplace_back(std::move(str));
    }
    if (i == 2) {
      std::cout << str;
    }

In this case, the check will erroneously produce a warning, even though it is
not possible for both the move and the use to be executed.

An erroneous warning can be silenced by reinitializing the object after the
move:

  .. code-block:: c++

    if (i == 1) {
      messages.emplace_back(std::move(str));
      str = "";
    }
    if (i == 2) {
      std::cout << str;
    }

Subsections below explain more precisely what exactly the check considers to be
a move, use, and reinitialization.

Unsequenced moves, uses, and reinitializations
----------------------------------------------

In many cases, C++ does not make any guarantees about the order in which
sub-expressions of a statement are evaluated. This means that in code like the
following, it is not guaranteed whether the use will happen before or after the
move:

  .. code-block:: c++

    void f(int i, std::vector<int> v);
    std::vector<int> v = { 1, 2, 3 };
    f(v[1], std::move(v));

In this kind of situation, the check will note that the use and move are
unsequenced.

The check will also take sequencing rules into account when reinitializations
occur in the same statement as moves or uses. A reinitialization is only
considered to reinitialize a variable if it is guaranteed to be evaluated after
the move and before the use.

Move
----

The check currently only considers calls of ``std::move`` on local variables or
function parameters. It does not check moves of member variables or global
variables.

Any call of ``std::move`` on a variable is considered to cause a move of that
variable, even if the result of ``std::move`` is not passed to an rvalue
reference parameter.

This means that the check will flag a use-after-move even on a type that does
not define a move constructor or move assignment operator. This is intentional.
Developers may use ``std::move`` on such a type in the expectation that the type
will add move semantics in the future. If such a ``std::move`` has the potential
to cause a use-after-move, we want to warn about it even if the type does not
implement move semantics yet.

Furthermore, if the result of ``std::move`` *is* passed to an rvalue reference
parameter, this will always be considered to cause a move, even if the function
that consumes this parameter does not move from it, or if it does so only
conditionally. For example, in the following situation, the check will assume
that a move always takes place:

  .. code-block:: c++

    std::vector<std::string> messages;
    void f(std::string &&str) {
      // Only remember the message if it isn't empty.
      if (!str.empty()) {
        messages.emplace_back(std::move(str));
      }
    }
    std::string str = "";
    f(std::move(str));

The check will assume that the last line causes a move, even though, in this
particular case, it does not. Again, this is intentional.

When analyzing the order in which moves, uses and reinitializations happen (see
section `Unsequenced moves, uses, and reinitializations`_), the move is assumed
to occur in whichever function the result of the ``std::move`` is passed to.

Use
---

Any occurrence of the moved variable that is not a reinitialization (see below)
is considered to be a use.

An exception to this are objects of type ``std::unique_ptr``,
``std::shared_ptr`` and ``std::weak_ptr``, which have defined move behavior
(objects of these classes are guaranteed to be empty after they have been moved
from). Therefore, an object of these classes will only be considered to be used
if it is dereferenced, i.e. if ``operator*``, ``operator->`` or ``operator[]``
(in the case of ``std::unique_ptr<T []>``) is called on it.

If multiple uses occur after a move, only the first of these is flagged.

Reinitialization
----------------

The check considers a variable to be reinitialized in the following cases:

  - The variable occurs on the left-hand side of an assignment.

  - The variable is passed to a function as a non-const pointer or non-const
    lvalue reference. (It is assumed that the variable may be an out-parameter
    for the function.)

  - ``clear()`` or ``assign()`` is called on the variable and the variable is of
    one of the standard container types ``basic_string``, ``vector``, ``deque``,
    ``forward_list``, ``list``, ``set``, ``map``, ``multiset``, ``multimap``,
    ``unordered_set``, ``unordered_map``, ``unordered_multiset``,
    ``unordered_multimap``.

  - ``reset()`` is called on the variable and the variable is of type
    ``std::unique_ptr``, ``std::shared_ptr`` or ``std::weak_ptr``.

If the variable in question is a struct and an individual member variable of
that struct is written to, the check does not consider this to be a
reinitialization -- even if, eventually, all member variables of the struct are
written to. For example:

  .. code-block:: c++

    struct S {
      std::string str;
      int i;
    };
    S s = { "Hello, world!\n", 42 };
    S s_other = std::move(s);
    s.str = "Lorem ipsum";
    s.i = 99;

The check will not consider ``s`` to be reinitialized after the last line;
instead, the line that assigns to ``s.str`` will be flagged as a use-after-move.
This is intentional as this pattern of reinitializing a struct is error-prone.
For example, if an additional member variable is added to ``S``, it is easy to
forget to add the reinitialization for this additional member. Instead, it is
safer to assign to the entire struct in one go, and this will also avoid the
use-after-move warning.
