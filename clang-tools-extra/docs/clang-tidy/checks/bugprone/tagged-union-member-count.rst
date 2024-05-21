.. title:: clang-tidy - bugprone-tagged-union-member-count

bugprone-tagged-union-member-count
==================================

Gives warnings for tagged unions, where the number of tags is
different from the number of data members inside the union.

A struct or a class is considered to be a tagged union if it has
exactly one union data member and exactly one enum data member and
any number of other data members that are neither unions or enums.

Example
-------

.. code-block:: c++

  enum tags {
    tag1,
    tag2,
  };

  struct taggedUnion { // warning: Tagged union has more data members (3) than tags (2)! [bugprone-tagged-union-member-count]
    enum tags kind;
    union {
      int i;
      float f;
      char *str;
    } data;
  };

Counting enum constant heuristic
--------------------------------

Some programmers use enums in such a way, where the value of the last enum 
constant is used to keep track of how many enum constants have been declared.

.. code-block:: c++

  enum tags_with_counter {
    tag1, // is 0
    tag2, // is 1
    tag3, // is 2
    tags_count, // is 3
  };

The check detects this usage pattern heuristically and does not include
the counter enum constant in the final tag count, since the counter is not
meant to indicate the valid variant member.

Options
-------

.. option:: EnumCounterHeuristicIsEnabled

    This option enables or disables the counting enum heuristic.
    To find possible counting enum constants this option uses the value of
    the string :option:`EnumCounterSuffix`.

    This option is enabled by default.

Example of :option:`EnumCounterHeuristicIsEnabled`:

When :option:`EnumCounterHeuristicIsEnabled` is false:

.. code-block:: c++

  enum tags_with_counter {
    tag1,
    tag2,
    tag3,
    tags_count,
  };

  struct taggedUnion {
    tags_with_counter tag;
    union data {
      int a;
      int b;
      char *str;
      float f;
    };
  };
 
When :option:`EnumCounterHeuristicIsEnabled` is true:

.. code-block:: c++

  enum tags_with_counter {
    tag1,
    tag2,
    tag3,
    tags_count,
  };

  struct taggedUnion { // warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
    tags_with_counter tag;
    union data {
      int a;
      int b;
      char *str;
      float f;
    };
  };

.. option:: EnumCounterSuffix

    When defined, the check will use the given string to search for counting
    enum constants. This option does not alter the check's behavior when
    :option:`EnumCounterHeuristicIsEnabled` is disabled.

    The default value is "count".

Example of :option:`EnumCounterSuffix`:

When :option:`EnumCounterHeuristicIsEnabled` is true and
:option:`EnumCounterSuffix` is "size":

.. code-block:: c++

  enum tags_with_counter_count {
    tag1,
    tag2,
    tag3,
    tags_count,
  };

  enum tags_with_counter_size {
    tag4,
    tag5,
    tag6,
    tags_size,
  };

  struct taggedUnion1 {
    tags_with_counter_count tag;
    union data {
      int a;
      int b;
      char *str;
      float f;
    };
  };

  struct taggedUnion2 { // warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
    tags_with_counter_size tag;
    union data {
      int a;
      int b;
      char *str;
      float f;
    };
  };
 
When :option:`EnumCounterSuffix` is true:

.. code-block:: c++

  enum tags_with_counter {
    tag1,
    tag2,
    tag3,
    tags_count,
  };

  struct taggedUnion { // warning: Tagged union has more data members (4) than tags (3)! [bugprone-tagged-union-member-count]
    tags_with_counter tag;
    union data {
      int a;
      int b;
      char *str;
      float f;
    };
  };

.. option:: StrictMode

    When enabled, the check will also give a warning, when the number of tags
    is greater than the number of union data members.

    This option is not enabled by default.

Example of :option:`StrictMode`:

When :option:`StrictMode` is false:

.. code-block:: c++

    struct taggedUnion {
      enum {
        tags1,
        tags2,
        tags3,
      } tags;
      union {
        int i;
        float f;
      };
    };

When :option:`StrictMode` is true:

.. code-block:: c++

    struct taggedunion1 { // warning: Tagged union has fewer data members (2) than tags (3)! [bugprone-tagged-union-member-count]
      enum {
        tags1,
        tags2,
        tags3,
      } tags;
      union {
        int i;
        float f;
      };
    };

