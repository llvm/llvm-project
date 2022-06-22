// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: readability-identifier-naming.ParameterCase, value: lower_case} \
// RUN:   ]}'

int func(int Break) {
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for parameter 'Break'; cannot be fixed because 'break' would conflict with a keyword
  // CHECK-FIXES: {{^}}int func(int Break) {{{$}}
  if (Break == 1) {
    // CHECK-FIXES: {{^}}  if (Break == 1) {{{$}}
    return 2;
  }

  return 0;
}

#define foo 3
int func2(int Foo) {
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for parameter 'Foo'; cannot be fixed because 'foo' would conflict with a macro definition
  // CHECK-FIXES: {{^}}int func2(int Foo) {{{$}}
  if (Foo == 1) {
    // CHECK-FIXES: {{^}}  if (Foo == 1) {{{$}}
    return 2;
  }

  return 0;
}

int func3(int _0Bad) {
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for parameter '_0Bad'; cannot be fixed automatically [readability-identifier-naming]
  // CHECK-FIXES: {{^}}int func3(int _0Bad) {{{$}}
  if (_0Bad == 1) {
    // CHECK-FIXES: {{^}}  if (_0Bad == 1) {{{$}}
    return 2;
  }
  return 0;
}
