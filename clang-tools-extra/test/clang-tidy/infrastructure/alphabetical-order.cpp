// RUN: %python %S/../../../clang-tidy/tool/clang-tidy-alphabetical-order-check.py checks-list -i %S/../../../docs/clang-tidy/checks/list.rst | diff --strip-trailing-cr - %S/../../../docs/clang-tidy/checks/list.rst

// RUN: %python %S/../../../clang-tidy/tool/clang-tidy-alphabetical-order-check.py release-notes -i %S/../../../docs/ReleaseNotes.rst | diff --strip-trailing-cr - %S/../../../docs/ReleaseNotes.rst
