// RUN: cp %S/Inputs/macro-to-enum/crlf.cpp %t.cpp
// RUN: chmod u+w %t.cpp
// RUN: clang-tidy %t.cpp -fix --checks='-*,modernize-macro-to-enum' --config={} -- --std=c++14 > %t.out 2>&1
// RUN: diff %t.cpp %S/Inputs/macro-to-enum/crlf.cpp.expected
