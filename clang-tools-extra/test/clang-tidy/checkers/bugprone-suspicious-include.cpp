// RUN: %check_clang_tidy %s bugprone-suspicious-include %t -- -- -isystem %S/Inputs/Headers -fmodules

// clang-format off

// CHECK-MESSAGES: [[@LINE+4]]:11: warning: suspicious #include of file with '.cpp' extension
// CHECK-MESSAGES: [[@LINE+3]]:11: note: did you mean to include 'a'?
// CHECK-MESSAGES: [[@LINE+2]]:11: note: did you mean to include 'a.h'?
// CHECK-MESSAGES: [[@LINE+1]]:11: note: did you mean to include 'a.hpp'?
#include "a.cpp"

// CHECK-MESSAGES: [[@LINE+2]]:11: warning: suspicious #include of file with '.cpp' extension
// CHECK-MESSAGES: [[@LINE+1]]:11: note: did you mean to include 'i.h'?
#include "i.cpp"

#include "b.h"

// CHECK-MESSAGES: [[@LINE+1]]:16: warning: suspicious #include_next of file with '.c' extension
#include_next <c.c>

// CHECK-MESSAGES: [[@LINE+1]]:13: warning: suspicious #include of file with '.cc' extension
# include  <c.cc>

// CHECK-MESSAGES: [[@LINE+1]]:14: warning: suspicious #include of file with '.cxx' extension
#  include  <c.cxx>
