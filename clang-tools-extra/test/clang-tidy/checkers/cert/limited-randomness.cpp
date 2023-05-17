// RUN: %check_clang_tidy %s cert-msc50-cpp %t

int rand();
int rand(int);

namespace std {
using ::rand;
}

namespace nonstd {
  int rand();
}

void testFunction1() {
  int i = std::rand();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: rand() has limited randomness; use C++11 random library instead [cert-msc50-cpp]

  int j = ::rand();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: rand() has limited randomness; use C++11 random library instead [cert-msc50-cpp]

  int k = rand(i);

  int l = nonstd::rand();

  int m = rand();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: rand() has limited randomness; use C++11 random library instead [cert-msc50-cpp]
}

