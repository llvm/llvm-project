.. title:: clang-tidy - readability-suspicious-call-argument

readability-suspicious-call-argument
====================================

Finds function calls where the arguments passed are provided out of order,
based on the difference between the argument name and the parameter names
of the function.

Given a function call ``f(foo, bar);`` and a function signature
``void f(T tvar, U uvar)``, the arguments ``foo`` and ``bar`` are swapped if
``foo`` (the argument name) is more similar to ``uvar`` (the other parameter)
than ``tvar`` (the parameter it is currently passed to) **and** ``bar`` is
more similar to ``tvar`` than ``uvar``.

Warnings might indicate either that the arguments are swapped, or that the
names' cross-similarity might hinder code comprehension.

.. _heuristics:

Heuristics
----------

The following heuristics are implemented in the check.
If **any** of the enabled heuristics deem the arguments to be provided out of
order, a warning will be issued.

The heuristics themselves are implemented by considering pairs of strings, and
are symmetric, so in the following there is no distinction on which string is
the argument name and which string is the parameter name.

Equality
^^^^^^^^

The most trivial heuristic, which compares the two strings for case-insensitive
equality.

.. _abbreviation_heuristic:

Abbreviation
^^^^^^^^^^^^

Common abbreviations can be specified which will deem the strings similar if
the abbreviated and the abbreviation stand together.
For example, if ``src`` is registered as an abbreviation for ``source``, then
the following code example will be warned about.

.. code-block:: c++

    void foo(int source, int x);

    foo(b, src);

The abbreviations to recognise can be configured with the
:ref:`Abbreviations<opt_Abbreviations>` check option.
This heuristic is case-insensitive.

Prefix
^^^^^^

The *prefix* heuristic reports if one of the strings is a sufficiently long
prefix of the other string, e.g. ``target`` to ``targetPtr``.
The similarity percentage is the length ratio of the prefix to the longer
string, in the previous example, it would be `6 / 9 = 66.66...`\%.

This heuristic can be configured with :ref:`bounds<opt_Bounds>`.
The default bounds are: below `25`\% dissimilar and above `30`\% similar.
This heuristic is case-insensitive.

Suffix
^^^^^^

Analogous to the `Prefix` heuristic.
In the case of ``oldValue`` and ``value`` compared, the similarity percentage
is `8 / 5 = 62.5`\%.

This heuristic can be configured with :ref:`bounds<opt_Bounds>`.
The default bounds are: below `25`\% dissimilar and above `30`\% similar.
This heuristic is case-insensitive.

Substring
^^^^^^^^^

The substring heuristic combines the prefix and the suffix heuristic, and tries
to find the *longest common substring* in the two strings provided.
The similarity percentage is the ratio of the found longest common substring
against the *longer* of the two input strings.
For example, given ``val`` and ``rvalue``, the similarity is `3 / 6 = 50`\%.
If no characters are common in the two string, `0`\%.

This heuristic can be configured with :ref:`bounds<opt_Bounds>`.
The default bounds are: below `40`\% dissimilar and above `50`\% similar.
This heuristic is case-insensitive.

Levenshtein distance (as `Levenshtein`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Levenshtein distance <http://en.wikipedia.org/wiki/Levenshtein_distance>`_
describes how many single-character changes (additions, changes, or removals)
must be applied to transform one string into another.

The Levenshtein distance is translated into a similarity percentage by dividing
it with the length of the *longer* string, and taking its complement with
regards to `100`\%.
For example, given ``something`` and ``anything``, the distance is `4` edits,
and the similarity percentage is `100`\% `- 4 / 9 = 55.55...`\%.

This heuristic can be configured with :ref:`bounds<opt_Bounds>`.
The default bounds are: below `50`\% dissimilar and above `66`\% similar.
This heuristic is case-sensitive.

Jaro–Winkler distance (as `JaroWinkler`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Jaro–Winkler distance <http://en.wikipedia.org/wiki/Jaro–Winkler_distance>`_
is an edit distance like the Levenshtein distance.
It is calculated from the amount of common characters that are sufficiently
close to each other in position, and to-be-changed characters.
The original definition of Jaro has been extended by Winkler to weigh prefix
similarities more.
The similarity percentage is expressed as an average of the common and
non-common characters against the length of both strings.

This heuristic can be configured with :ref:`bounds<opt_Bounds>`.
The default bounds are: below `75`\% dissimilar and above `85`\% similar.
This heuristic is case-insensitive.

Sørensen–Dice coefficient (as `Dice`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Sørensen–Dice coefficient <http://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
was originally defined to measure the similarity of two sets.
Formally, the coefficient is calculated by dividing `2 * #(intersection)` with
`#(set1) + #(set2)`, where `#()` is the cardinality function of sets.
This metric is applied to strings by creating bigrams (substring sequences of
length 2) of the two strings and using the set of bigrams for the two strings
as the two sets.

This heuristic can be configured with :ref:`bounds<opt_Bounds>`.
The default bounds are: below `60`\% dissimilar and above `70`\% similar.
This heuristic is case-insensitive.


Options
-------

.. option:: MinimumIdentifierNameLength

    Sets the minimum required length the argument and parameter names
    need to have. Names shorter than this length will be ignored.
    Defaults to `3`.

.. _opt_Abbreviations:

.. option:: Abbreviations

    For the **Abbreviation** heuristic
    (:ref:`see here<abbreviation_heuristic>`), this option configures the
    abbreviations in the `"abbreviation=abbreviated_value"` format.
    The option is a string, with each value joined by `";"`.

    By default, the following abbreviations are set:

       * `addr=address`
       * `arr=array`
       * `attr=attribute`
       * `buf=buffer`
       * `cl=client`
       * `cnt=count`
       * `col=column`
       * `cpy=copy`
       * `dest=destination`
       * `dist=distance`
       * `dst=distance`
       * `elem=element`
       * `hght=height`
       * `i=index`
       * `idx=index`
       * `len=length`
       * `ln=line`
       * `lst=list`
       * `nr=number`
       * `num=number`
       * `pos=position`
       * `ptr=pointer`
       * `ref=reference`
       * `src=source`
       * `srv=server`
       * `stmt=statement`
       * `str=string`
       * `val=value`
       * `var=variable`
       * `vec=vector`
       * `wdth=width`

The configuration options for each implemented heuristic (see above) is
constructed dynamically.
In the following, `<HeuristicName>` refers to one of the keys from the
heuristics implemented.

.. option:: <HeuristicName>

    `True` or `False`, whether a particular heuristic, such as `Equality` or
    `Levenshtein` is enabled.

    Defaults to `True` for every heuristic.

.. _opt_Bounds:

.. option:: <HeuristicName>DissimilarBelow, <HeuristicName>SimilarAbove

    A value between `0` and `100`, expressing a percentage.
    The bounds set what percentage of similarity the heuristic must deduce
    for the two identifiers to be considered similar or dissimilar by the
    check.

    Given arguments ``arg1`` and ``arg2`` passed to ``param1`` and ``param2``,
    respectively, the bounds check is performed in the following way:
    If the similarity of the currently passed argument order
    (``arg1`` to ``param1``) is **below** the `DissimilarBelow` threshold, and
    the similarity of the suggested swapped order (``arg1`` to ``param2``) is
    **above** the `SimilarAbove` threshold, the swap is reported.

    For the defaults of each heuristic, :ref:`see above<heuristics>`.


Name synthesis
--------------

When comparing the argument names and parameter names, the following logic is
used to gather the names for comparison:

Parameter names are the identifiers as written in the source code.

Argument names are:

  * If a variable is passed, the variable's name.
  * If a subsequent function call's return value is used as argument, the called
    function's name.
  * Otherwise, empty string.

Empty argument or parameter names are ignored by the heuristics.
