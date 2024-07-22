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

  struct TaggedUnion { // warning: tagged union has more data members (3) than tags (2)
    enum Tags Kind;
    union {
      int I;
      float F;
      char *Str;
    } Data;
  };

Counting enum constant heuristic
================================

Sometimes the last enum constant in an enum is used to keep track of how many
enum constants have been declared. For an illustration:

.. code-block:: c++

  enum TagWithCounter {
    Tag1, // is 0
    Tag2, // is 1
    Tag3, // is 2
    TagCount, // is 3
  };

When an enum like this is used as the tag for a tagged union then the last
"counting" enum constant will intentionally not have a corresponding union
data member.

This usage pattern is detected heuristically and the check
does not include the counter enum constant in the final tag count.
If the heuristic can be applied to multiple enum constants, then the check will
just count the enum constants normally and not modify the final count.

This heuristic can be disabled entirely (:option:`EnableCountingEnumHeuristic`) or
configured to follow your naming convention (:option:`CountingEnumPrefixes/Suffixes`).
The strings specified in (:option:`CountingEnumPrefixes/Suffixes`) are matched
case insensitively.

Options
=======

.. option:: EnableCountingEnumHeuristic

This option enables or disables the counting enum heuristic.
It uses the prefixes and suffixes specified in the options
:option:`CountingEnumPrefixes/Suffixes` to find counting enum constants by
using them for prefix and suffix matching.

This option is enabled by default.

When :option:`EnableCountingEnumHeuristic` is false:

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
 
When :option:`EnableCountingEnumHeuristic` is true:

.. code-block:: c++

  enum TagWithCounter {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnion { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounter Kind;
    union Data {
      int A;
      long B;
      char *Str;
      float F;
    };
  };

.. option:: CountingEnumPrefixes/Suffixes

CountingEnumPrefixes and CountingEnumSuffixes are lists of semicolon
separated strings that are used to search for possible counting enum constants.
These strings are matched case insensitively as prefixes and suffixes
respectively on the names of the enum constants.
If :option:`EnableCountingEnumHeuristic` is false then these options do nothing.

The default value of CountingEnumSuffixes is "count" and of
CountingEnumPrefixes is "" (empty string).

When :option:`EnableCountingEnumHeuristic` is true and CountingEnumSuffixes
is "count;size":

.. code-block:: c++

  enum TagWithCounterCount {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnionCount { // warning: tagged union has more data members (4) than tags (3)
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

  struct TaggedUnionSize { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterSize Kind;
    union Data {
      int A;
      long B;
      char *Str;
      float F;
    };
  };

When :option:`EnableCountingEnumHeuristic` is true and CountingEnumPrefixes is "maxsize;last_"

.. code-block:: c++

  enum TagWithCounterLast {
    Tag1,
    Tag2,
    Tag3,
    last_tag,
  };

  struct TaggedUnionLast { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterLast tag;
    union Data {
      int I;
      short S;
      char *C;
      float F;
    };
  };

  enum TagWithCounterMaxSize {
    Tag1,
    Tag2,
    Tag3,
    MaxSizeTag,
  };

  struct TaggedUnionMaxSize { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterMaxSize tag;
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

    struct TaggedUnion { // warning: tagged union has fewer data members (2) than tags (3)
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
