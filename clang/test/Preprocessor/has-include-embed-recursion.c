// RUN: %python -c "print('#if ' + '__has_include(' * 4096 + '0' + ')' * 4096 + '\n#endif')" > %t.include.c
// RUN: not %clang_cc1 -Eonly -std=c23 -ferror-limit 1 %t.include.c 2>&1 | FileCheck %s --check-prefix=INCLUDE
// RUN: not %clang_cc1 -Eonly -x c++ -std=c++20 -ferror-limit 1 %t.include.c 2>&1 | FileCheck %s --check-prefix=INCLUDE
// RUN: %python -c "print('#if ' + '__has_embed(' * 4096 + '0' + ')' * 4096 + '\n#endif')" > %t.embed.c
// RUN: not %clang_cc1 -Eonly -std=c23 -ferror-limit 1 %t.embed.c 2>&1 | FileCheck %s --check-prefix=EMBED
// RUN: not %clang_cc1 -Eonly -x c++ -std=c++20 -ferror-limit 1 %t.embed.c 2>&1 | FileCheck %s --check-prefix=EMBED

// Diagnose the third builtin without recursively expanding the rest. Macro
// expansion disables header-name tokenization, but must not disable the
// protection against reentering LexIncludeFilename.
// INCLUDE: :1:33: error: expected "FILENAME" or <FILENAME>
// EMBED: :1:29: error: expected "FILENAME" or <FILENAME>
