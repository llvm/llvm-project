.. title:: clang-tidy - bugprone-tagged-union-member-count

bugprone-tagged-union-member-count
==================================

Gives warnings for tagged unions, where the number of tags is
different from the number of data members inside the union.

A struct or a class is considered to be a tagged union if it has
exactly one union data member and exactly one enum data member and
any number of other data members that are neither unions or enums.

Example:

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

How enum constants are counted
------------------------------

The main complicating factor when counting the number of enum constants is that
some of them might be auxiliary values that purposefully don't have a corresponding union
data member and are used for something else. For example the last enum constant
sometimes explicitly "points to" the last declared valid enum constant or
tracks how many enum constants have been declared.

For an illustration:

.. code-block:: c++

  enum TagWithLast {
    Tag1 = 0,
    Tag2 = 1,
    Tag3 = 2,
    LastTag = 2
  };

  enum TagWithCounter {
    Tag1, // is 0
    Tag2, // is 1
    Tag3, // is 2
    TagCount, // is 3
  };

The check counts the number of distinct values among the enum constants and not the enum
constants themselves. This way the enum constants that are essentially just aliases of other
enum constants are not included in the final count.

Handling of counting enum constants (ones like :code:`TagCount` in the previous code example)
is done by decreasing the number of enum values by one if the name of the last enum constant
starts with a prefix or ends with a suffix specified in :option:`CountingEnumPrefixes`,
:option:`CountingEnumSuffixes` and it's value is one less than the total number of distinct
values in the enum.

When the final count is adjusted based on this heuristic then a diagnostic note is emitted
that shows which enum constant matched the criteria.

The heuristic can be disabled entirely (:option:`EnableCountingEnumHeuristic`) or
configured to follow your naming convention (:option:`CountingEnumPrefixes`, :option:`CountingEnumSuffixes`).
The strings specified in :option:`CountingEnumPrefixes`, :option:`CountingEnumSuffixes` are matched
case insensitively.

Example counts:

.. code-block:: c++

  // Enum count is 3, because the value 2 is counted only once
  enum TagWithLast {
    Tag1 = 0,
    Tag2 = 1,
    Tag3 = 2,
    LastTag = 2
  };

  // Enum count is 3, because TagCount is heuristically excluded
  enum TagWithCounter {
    Tag1, // is 0
    Tag2, // is 1
    Tag3, // is 2
    TagCount, // is 3
  };


Options
-------

.. option:: EnableCountingEnumHeuristic

This option enables or disables the counting enum heuristic.
It uses the prefixes and suffixes specified in the options
:option:`CountingEnumPrefixes`, :option:`CountingEnumSuffixes` to find counting enum constants by
using them for prefix and suffix matching.

This option is enabled by default.

When :option:`EnableCountingEnumHeuristic` is `false`:

.. code-block:: c++

  enum TagWithCounter {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnion {
    TagWithCounter Kind;
    union {
      int A;
      long B;
      char *Str;
      float F;
    } Data;
  };

When :option:`EnableCountingEnumHeuristic` is `true`:

.. code-block:: c++

  enum TagWithCounter {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnion { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounter Kind;
    union {
      int A;
      long B;
      char *Str;
      float F;
    } Data;
  };

.. option:: CountingEnumPrefixes

See :option:`CountingEnumSuffixes` below.

.. option:: CountingEnumSuffixes

CountingEnumPrefixes and CountingEnumSuffixes are lists of semicolon
separated strings that are used to search for possible counting enum constants.
These strings are matched case insensitively as prefixes and suffixes
respectively on the names of the enum constants.
If :option:`EnableCountingEnumHeuristic` is `false` then these options do nothing.

The default value of :option:`CountingEnumSuffixes` is `count` and of
:option:`CountingEnumPrefixes` is the empty string.

When :option:`EnableCountingEnumHeuristic` is `true` and :option:`CountingEnumSuffixes`
is `count;size`:

.. code-block:: c++

  enum TagWithCounterCount {
    Tag1,
    Tag2,
    Tag3,
    TagCount,
  };

  struct TaggedUnionCount { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterCount Kind;
    union {
      int A;
      long B;
      char *Str;
      float F;
    } Data;
  };

  enum TagWithCounterSize {
    Tag11,
    Tag22,
    Tag33,
    TagSize,
  };

  struct TaggedUnionSize { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterSize Kind;
    union {
      int A;
      long B;
      char *Str;
      float F;
    } Data;
  };

When :option:`EnableCountingEnumHeuristic` is `true` and :option:`CountingEnumPrefixes` is `maxsize;last_`

.. code-block:: c++

  enum TagWithCounterLast {
    Tag1,
    Tag2,
    Tag3,
    last_tag,
  };

  struct TaggedUnionLast { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterLast tag;
    union {
      int I;
      short S;
      char *C;
      float F;
    } Data;
  };

  enum TagWithCounterMaxSize {
    Tag1,
    Tag2,
    Tag3,
    MaxSizeTag,
  };

  struct TaggedUnionMaxSize { // warning: tagged union has more data members (4) than tags (3)
    TagWithCounterMaxSize tag;
    union {
      int I;
      short S;
      char *C;
      float F;
    } Data;
  };

.. option:: StrictMode

When enabled, the check will also give a warning, when the number of tags
is greater than the number of union data members.

This option is disabled by default.

When :option:`StrictMode` is `false`:

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
      } Data;
    };

When :option:`StrictMode` is `true`:

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
      } Data;
    };
