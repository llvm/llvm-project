.. title:: clang-tidy - modernize-use-std-erase

modernize-use-std-erase
=======================

Replaces erase-remove idiom with C++20's' std::erase and std::erase_if 
for improved readability.

Covered scenarios:

========================================================== ============================
Expression                                                 Replacement
---------------------------------------------------------- ----------------------------
``v.erase(std::remove(v.begin(), v.end(), 5), v.end())``   ``std::erase(v, 5)``
``l.erase(std::remove_if(v.begin(), v.end(), 5), isEven)`` ``std::erase_if(v, isEven)``
========================================================== ============================
