// RUN: %clang_cc1 -E -ftrigraphs %s | grep -- ' ->'
// RUN: %clang_cc1 -E -ftrigraphs %s 2>&1 | grep 'backslash and newline separated by space'
// RUN: %clang_cc1 -E -ftrigraphs %s 2>&1 | grep 'trigraph converted'
// RUN: %clang_cc1 -E -CC -ftrigraphs %s

// This is an ugly way to spell a -> token.
 -??/      
>

// \

