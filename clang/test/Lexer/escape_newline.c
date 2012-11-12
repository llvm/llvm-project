// RUN: %clang_cc1 -E -trigraphs %s | grep -- ' ->'
// RUN: %clang_cc1 -E -trigraphs %s 2>&1 | grep 'backslash and newline separated by space'
// RUN: %clang_cc1 -E -trigraphs %s 2>&1 | grep 'trigraph converted'
// RUN: %clang_cc1 -E -CC -trigraphs %s

// This is an ugly way to spell a -> token.
 -??/      
>

// \

