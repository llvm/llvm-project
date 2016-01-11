.. title:: clang-tidy - misc-throw-by-value-catch-by-reference

misc-throw-by-value-catch-by-reference
======================================

"cert-err61-cpp" redirects here as an alias for this checker.

Finds violations of the rule "Throw by value, catch by reference" presented for example in "C++ Coding Standards" by H. Sutter and A. Alexandrescu. This check also has the option to find violations of the rule "Throw anonymous temporaries" (https://www.securecoding.cert.org/confluence/display/cplusplus/ERR09-CPP.+Throw+anonymous+temporaries). The option is named "CheckThrowTemporaries" and it's on by default.

Exceptions:
- throwing string literals will not be flagged despite being a pointer. They are not susceptible to slicing and the usage of string literals is idomatic.
- catching character pointers (char, wchar_t, unicode character types) will not be flagged to allow catching sting literals.
- moved named values will not be flagged as not throwing an anonymous temporary. In this case we can be sure that the user knows that the object can't be accessed outside catch blocks handling the error.
- throwing function parameters will not be flagged as not throwing an anonymous temporary. This allows helper functions for throwing.
- re-throwing caught exception variables will not be flragged as not throwing an anonymous temporary. Although this can usually be done by just writing "throw;" it happens often enough in real code.
