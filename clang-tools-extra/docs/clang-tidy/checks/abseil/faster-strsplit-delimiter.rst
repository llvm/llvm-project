.. title:: clang-tidy - abseil-faster-strsplit-delimiter

abseil-faster-strsplit-delimiter
================================

Finds instances of ``absl::StrSplit()`` or ``absl::MaxSplits()`` where the
delimiter is a single character string literal and replaces with a character.
The check will offer a suggestion to change the string literal into a character.
It will also catch code using ``absl::ByAnyChar()`` for just a single character
and will transform that into a single character as well.

These changes will give the same result, but using characters rather than
single character string literals is more efficient and readable.

Examples:

.. code-block:: c++

  // Original - the argument is a string literal.
  for (auto piece : absl::StrSplit(str, "B")) {

  // Suggested - the argument is a character, which causes the more efficient
  // overload of absl::StrSplit() to be used.
  for (auto piece : absl::StrSplit(str, 'B')) {


  // Original - the argument is a string literal inside absl::ByAnyChar call.
  for (auto piece : absl::StrSplit(str, absl::ByAnyChar("B"))) {

  // Suggested - the argument is a character, which causes the more efficient
  // overload of absl::StrSplit() to be used and we do not need absl::ByAnyChar
  // anymore.
  for (auto piece : absl::StrSplit(str, 'B')) {


  // Original - the argument is a string literal inside absl::MaxSplits call.
  for (auto piece : absl::StrSplit(str, absl::MaxSplits("B", 1))) {

  // Suggested - the argument is a character, which causes the more efficient
  // overload of absl::StrSplit() to be used.
  for (auto piece : absl::StrSplit(str, absl::MaxSplits('B', 1))) {
