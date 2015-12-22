.. title:: clang-tidy - cppcoreguidelines-pro-type-union-access

cppcoreguidelines-pro-type-union-access
=======================================

This check flags all access to members of unions. Passing unions as a whole is not flagged.

Reading from a union member assumes that member was the last one written, and writing to a union member assumes another member with a nontrivial destructor had its destructor called. This is fragile because it cannot generally be enforced to be safe in the language and so relies on programmer discipline to get it right.

This rule is part of the "Type safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#-type7-avoid-accessing-members-of-raw-unions-prefer-variant-instead
