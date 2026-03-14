// RUN: %clang_cc1 -include %S/file_to_include.h -E %s -fno-caret-diagnostics 2>&1 >/dev/null | grep 'file successfully included' | count 1
// PR3464

