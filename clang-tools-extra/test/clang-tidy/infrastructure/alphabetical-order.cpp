// RUN: %python %S/../../../clang-tidy/tool/clang-tidy-alphabetical-order-check.py checks-list -i %S/../../../docs/clang-tidy/checks/list.rst -o %t.list
// RUN: diff --strip-trailing-cr %t.list \
// RUN:   %S/../../../docs/clang-tidy/checks/list.rst

// RUN: %python %S/../../../clang-tidy/tool/clang-tidy-alphabetical-order-check.py release-notes -i %S/../../../docs/ReleaseNotes.rst -o %t.rn
// RUN: diff --strip-trailing-cr %t.rn %S/../../../docs/ReleaseNotes.rst
