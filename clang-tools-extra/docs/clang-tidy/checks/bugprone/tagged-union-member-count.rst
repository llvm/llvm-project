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

  enum tags2 {
    tag1,
    tag2,
  };

  struct taggedunion { // warning: Tagged union has more data members than tags! Data members: 3 Tags: 2 [bugprone-tagged-union-member-count]
    enum tags2 kind;
    union {
      int i;
      float f;
      char *str;
    } data;
  };

  enum tags4 {
    tag1,
    tag2,
    tag3,
    tag4,
  };
  
  struct taggedunion { // warning: Tagged union has more fewer members than tags! Data members: 3 Tags: 4 [bugprone-tagged-union-member-count]
    enum tags4 kind;
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

The checker detects this usage pattern heuristically and does not include
the counter enum constant in the final tag count, since the counter is not
meant to indicate the valid variant member.

