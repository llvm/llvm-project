// RUN: %clang_cc1 -E -verify %s

#pragma clang glibcxx_version // expected-error {{expected integer after '#pragma clang glibcxx_version'}}
#pragma clang glibcxx_version foo // expected-error {{expected integer after '#pragma clang glibcxx_version'}}
#pragma clang glibcxx_version 100000000000000000000000000000 // expected-error {{expected integer after '#pragma clang glibcxx_version'}}
