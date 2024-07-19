.. title:: clang-tidy - bugprone-tagged-union-member-count

==================================
bugprone-tagged-union-member-count
==================================

Gives warnings for tagged unions, where the number of tags is
different from the number of data members inside the union.

A struct or a class is considered to be a tagged union if it has
exactly one union data member and exactly one enum data member and
any number of other data members that are neither unions or enums.

Example
=======

.. code-block:: c++

  enum Tags {
    Tag1,
    Tag2,
  };

  struct TaggedUnion { // warning: Tagged union has more data members (3) than tags (2)
    enum Tags Kind;
    union {
      int I;
      float F;
      char *Str;
    } Data;
  };

Counting enum constant heuristic
================================

Some programmers use enums in such a way, where the value of the last enum 
constant is used to keep track of how many enum constants have been declared.

.. code-block:: c++

  enum TagWithCounter {
    Tag1, // is 0
    Tag2, // is 1
    Tag3, // is 2
    TagCount, // is 3
  };

This usage pattern is detected heuristically and the check does not include
the counter enum constant in the final tag count, since the counter is not
meant to indicate the valid union data member.

When the check finds multiple possible counting enums, then it does not change the enum count.

This heuristic can be disabled entirely (:option:`CountingEnumHeuristicIsEnabled`) or
configured to follow your naming convention (:option:`CountingEnumPrefixes/Suffixes`).
String matching is done case insensitively.

Options
=======

.. option:: CountingEnumHeuristicIsEnabled

This option enables or disables the counting enum heuristic.
To find possible counting enum constants this option uses the prefixes
and suffixes specified 
the string :option:`CountingEnumSuffixes`.

This option is enabled by default.

When :option:`CountingEnumHeuristicIsEnabled` is false:

.. code-block:: c++

  enum TagWithCounter {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnion {
    TagWithCounter Kind;
    union Data {
      int A;
      long B;
      char *Str;
      float F;
    };
  };
 
When :option:`CountingEnumHeuristicIsEnabled` is true:

.. code-block:: c++

  enum TagWithCounter {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnion { // warning: Tagged union has more data members (4) than tags (3)
    TagWithCounter Kind;
    union Data {
      int A;
      long B;
      char *Str;
      float F;
    };
  };

.. option:: CountingEnumPrefixes/Suffixes

When defined, the check will use the list of the semicolon separated strings
in CountingEnumPrefixes or CountingEnumSuffixes for the identification of possible counting enum constants.
These options do not alter the check's behavior when :option:`CountingEnumHeuristicIsEnabled` is set to false.

The default value for CountingEnumSuffixes is "count" and for CountingEnumPrefixes is "" (empty string).

When :option:`CountingEnumHeuristicIsEnabled` is true and CountingEnumSuffixes is "count;size":

.. code-block:: c++

  enum TagWithCounterCount {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnionCount { // warning: Tagged union has more data members (4) than tags (3)
    TagWithCounterCount Kind;
    union Data {
      int A;
      long B;
      char *Str;
      float F;
    };
  };

  enum TagWithCounterSize {
    Tag11,
    Tag22,
    Tag33,
    TagSize,
  };

  struct TaggedUnionSize { // warning: Tagged union has more data members (4) than tags (3)
    TagWithCounterSize Kind;
    union Data {
      int A;
      long B;
      char *Str;
      float F;
    };
  };

When :option:`CountingEnumHeuristicIsEnabled` is true and CountingEnumPrefixes is "maxsize;last_"

.. code-block:: c++

  enum TagWithCounter {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnion { // warning: Tagged union has more data members (4) than tags (3)
    TagWithCounter tag;
    union Data {
      int I;
      short S;
      char *C;
      float F;
    };
  };

.. option:: StrictMode

When enabled, the check will also give a warning, when the number of tags
is greater than the number of union data members.

This option is disabled by default.

When :option:`StrictMode` is false:

.. code-block:: c++

    struct TaggedUnion {
      enum {
        Tag1,
        Tag2,
        Tag3,
      } Tags;
      union {
        int I;
        float F;
      };
    };

When :option:`StrictMode` is true:

.. code-block:: c++

    struct TaggedUnion { // warning: Tagged union has fewer data members (2) than tags (3)
      enum {
        Tag1,
        Tag2,
        Tag3,
      } Tags;
      union {
        int I;
        float F;
      };
    };

