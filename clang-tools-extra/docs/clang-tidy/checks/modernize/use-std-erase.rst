.. title:: clang-tidy - modernize-use-std-erase

modernize-use-std-erase
=======================

Replaces erase-remove idiom with C++20's ``std::erase`` and ``std::erase_if``
for improved readability.

Covered scenarios:

========================================================= =====================
Expression                                                Replacement
--------------------------------------------------------- ---------------------
``v.erase(remove(v.begin(), v.end(), 5), v.end())``       ``erase(v, 5)``
``l.erase(remove_if(l.begin(), l.end(), func), l.end())`` ``erase_if(l, func)``
========================================================= =====================
