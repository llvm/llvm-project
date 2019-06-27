// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -std=c++11 %s 2>&1 | FileCheck %s

enum Color {
  Black,
  Blue,
  White,
  Gold
};

void fillInCases(Color c) {
  switch (c) {
  case Black:
    break;
  }
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"case Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n"
  switch (c) {
  }
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n"
}

enum class NoDenseMap: long long {
  Baddie = 0x7fffffffffffffffLL,
  BigBaddie = -0x7fffffffffffffffLL-1
};

void fillInAllCases(NoDenseMap v) {
  switch (v) {
  }
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"case NoDenseMap::Baddie:\n<#code#>\nbreak;\ncase NoDenseMap::BigBaddie:\n<#code#>\nbreak;\n"
}

