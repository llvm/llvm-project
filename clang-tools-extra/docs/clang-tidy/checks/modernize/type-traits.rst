.. title:: clang-tidy - modernize-type-traits

modernize-type-traits
=====================

Converts standard library type traits of the form ``traits<...>::type`` and
``traits<...>::value`` into ``traits_t<...>`` and
``traits_v<...>`` respectively.

For example:

.. code-block:: c++

  std::is_integral<T>::value
  std::is_same<int, float>::value
  typename std::add_const<T>::type
  std::make_signed<unsigned>::type

Would be converted into:

.. code-block:: c++

  std::is_integral_v<T>
  std::is_same_v<int, float>
  std::add_const_t<T>
  std::make_signed_t<unsigned>

Options
-------

.. option:: IgnoreMacros

  If `true` don't diagnose traits defined in macros.

  Note: Fixes will never be emitted for code inside of macros.

  .. code-block:: c++

    #define IS_SIGNED(T) std::is_signed<T>::value

  Defaults to `false`.


Limitations
-----------

Does not currently diagnose uses of type traits with nested name
specifiers (e.g. ``std::chrono::is_clock``,
``std::chrono::treat_as_floating_point``).


Quick-and-dirty migration script
--------------------------------

The following (hacky) Bash script can be used to migrate files en mass via Perl, with the obvious caveat that operates textually and thus may misunderstand complex code:

.. code-block:: shell

  #!/usr/bin/env bash

  symbols=($(\
    clang++ -D_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS -std=c++26 -E -P -stdlib=libc++ -include type_traits -x c++ /dev/null \
    | perl -0777 -n -l -e '$seen{$1}++ while />[^;{}]*(?:using|const)[^{};]*\b([a-z][a-z0-9_]+_[vt])\b[^=;{}]*=/g; END { print "$_\n" for sort keys %seen }'
  ))
  
  t="$(printf "%s|" "${symbols[@]}" | sed 's/\b[^|]*_v\b//; s/_t\b//g; s/|\+/|/g; s/^|\||$//g')"
  v="$(printf "%s|" "${symbols[@]}" | sed 's/\b[^|]*_t\b//; s/_v\b//g; s/|\+/|/g; s/^|\||$//g')"
  
  perl -0777 -i -p -e '
    BEGIN {
      our ($G, $P, $B, $C);
      $P = qr/(?>(?:[^""'\'\''()]+|\((??{$P})\))*)/;
      $B = qr/(?>(?:[^""'\'\''\[\]]+|\[(??{$B})\])*)/;
      $C = qr/(?>(?:[^""'\'\''<>]+|<(??{$C})>)*)/;
      $G = qr/(?>[^""'\'\''()\[\]<>]|\((??{$P})\)|\[(??{$B})\]|<(??{$C})>)*/;
    }
    1 while (
      s/(\b(?:typename\s+))((?<!\w)(?:(?:::\s*)?std\s*::\s*(?:'"${t}"')))(?<!_t)(\s*<(??{$G})>\s*)::\s*type\b(\s*::)?((?=>))?/
        (defined $4 ? $1 : "") . $2 . "_t" . $3 . $4 . (defined $5 ? " " : "")/ge
      or
      s/((?<!\w)(?:(?:::\s*)?std\s*::\s*'"${v}"'))(?<!_v)\s*<((??{$G}))>\s*::\s*value/$1_v<$2>/g
    )
  ' "$@"
