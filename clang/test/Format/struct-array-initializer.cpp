// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style="{BasedOnStyle: LLVM, AlignArrayOfStructures: Right}" %s \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK1 %s
// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style="{BasedOnStyle: LLVM, AlignArrayOfStructures: Left}" %s \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK2 %s
struct test {
  int a;
  int b;
  const char *c;
};

struct toast {
  int a;
  const char *b;
  int c;
  float d;
};

void f() {
  struct test demo[] = {{56, 23, "hello"}, {-1, 93463, "world"}, {7, 5, "!!"}};
  // CHECK1: {{^[[:space:]]{2}struct test demo\[\] = \{$}}
  // CHECK1-NEXT: {{([[:space:]]{4})}}{56,    23, "hello"},
  // CHECK1-NEXT: {{([[:space:]]{4})}}{-1, 93463, "world"},
  // CHECK1-NEXT: {{([[:space:]]{4})}}{ 7,     5,    "!!"}
  // CHECK1-NEXT: {{^[[:space:]]{2}\};$}}
}

void g() {
  struct toast demo[] = {
      {56, "hello world I have some things to say", 30, 4.2},
      {93463, "those things are really comments", 1, 3.1},
      {7, "about a wide range of topics", 789, .112233}};
  // CHECK1: {{^[[:space:]]{2}struct toast demo\[\] = \{$}}
  // CHECK1-NEXT: {{([[:space:]]{4})}}{   56, "hello world I have some things to say",  30,     4.2},
  // CHECK1-NEXT: {{([[:space:]]{4})}}{93463,      "those things are really comments",   1,     3.1},
  // CHECK1-NEXT: {{([[:space:]]{4})}}{    7,          "about a wide range of topics", 789, .112233}
  // CHECK1-NEXT: {{^[[:space:]]{2}\};$}}
}

void h() {
  struct test demo[] = {{56, 23, "hello"}, {-1, 93463, "world"}, {7, 5, "!!"}};
  // CHECK2: {{^[[:space:]]{2}struct test demo\[\] = \{$}}
  // CHECK2-NEXT: {{([[:space:]]{4})}}{56, 23,    "hello"},
  // CHECK2-NEXT: {{([[:space:]]{4})}}{-1, 93463, "world"},
  // CHECK2-NEXT: {{([[:space:]]{4})}}{7,  5,     "!!"   }
  // CHECK2-NEXT: {{^[[:space:]]{2}\};$}}
}

void i() {
  struct toast demo[] = {
      {56, "hello world I have some things to say", 30, 4.2},
      {93463, "those things are really comments", 1, 3.1},
      {7, "about a wide range of topics", 789, .112233}};
  // CHECK2: {{^[[:space:]]{2}struct toast demo\[\] = \{$}}
  // CHECK2-NEXT: {{([[:space:]]{4})}}{56,    "hello world I have some things to say", 30,  4.2    },
  // CHECK2-NEXT: {{([[:space:]]{4})}}{93463, "those things are really comments",      1,   3.1    },
  // CHECK2-NEXT: {{([[:space:]]{4})}}{7,     "about a wide range of topics",          789, .112233}
  // CHECK2-NEXT: {{^[[:space:]]{2}\};$}}
}
