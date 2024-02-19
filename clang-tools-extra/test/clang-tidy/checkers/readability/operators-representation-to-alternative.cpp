// RUN: %check_clang_tidy %s readability-operators-representation %t -- -config="{CheckOptions: {\
// RUN: readability-operators-representation.BinaryOperators: 'and;and_eq;bitand;bitor;compl;not;not_eq;or;or_eq;xor;xor_eq', \
// RUN: readability-operators-representation.OverloadedOperators: 'and;and_eq;bitand;bitor;compl;not;not_eq;or;or_eq;xor;xor_eq'}}" --

void testAllTokensToAlternative(int a, int b) {
  int value = 0;

  value = a||b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '||' is a traditional token spelling, consider using an alternative token 'or' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a or b;{{$}}

  value = a&&b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '&&' is a traditional token spelling, consider using an alternative token 'and' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a and b;{{$}}

  value = a | b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '|' is a traditional token spelling, consider using an alternative token 'bitor' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a bitor b;{{$}}

  value = a & b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '&' is a traditional token spelling, consider using an alternative token 'bitand' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a bitand b;{{$}}

  value = !a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '!' is a traditional token spelling, consider using an alternative token 'not' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = not a;{{$}}

  value = a^b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '^' is a traditional token spelling, consider using an alternative token 'xor' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a xor b;{{$}}

  value = ~b;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '~' is a traditional token spelling, consider using an alternative token 'compl' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = compl b;{{$}}

  value &= b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '&=' is a traditional token spelling, consider using an alternative token 'and_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value and_eq b;{{$}}

  value |= b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '|=' is a traditional token spelling, consider using an alternative token 'or_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value or_eq b;{{$}}

  value = a != b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '!=' is a traditional token spelling, consider using an alternative token 'not_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a not_eq b;{{$}}

  value ^= a;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '^=' is a traditional token spelling, consider using an alternative token 'xor_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value xor_eq a;{{$}}
}

struct Class {
  bool operator!() const;
  Class operator~() const;
  bool operator&&(const Class&) const;
  Class operator&(const Class&) const;
  bool operator||(const Class&) const;
  Class operator|(const Class&) const;
  Class operator^(const Class&) const;
  Class& operator&=(const Class&) const;
  Class& operator|=(const Class&) const;
  Class& operator^=(const Class&) const;
  bool operator!=(const Class&) const;
};

void testAllTokensToAlternative(Class a, Class b) {
  int value = 0;
  Class clval;

  value = a||b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '||' is a traditional token spelling, consider using an alternative token 'or' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a or b;{{$}}

  value = a&&b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '&&' is a traditional token spelling, consider using an alternative token 'and' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a and b;{{$}}

  clval = a | b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '|' is a traditional token spelling, consider using an alternative token 'bitor' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a bitor b;{{$}}

  clval = a & b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '&' is a traditional token spelling, consider using an alternative token 'bitand' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a bitand b;{{$}}

  value = !a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '!' is a traditional token spelling, consider using an alternative token 'not' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = not a;{{$}}

  clval = a^b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '^' is a traditional token spelling, consider using an alternative token 'xor' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a xor b;{{$}}

  clval = ~b;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '~' is a traditional token spelling, consider using an alternative token 'compl' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = compl b;{{$}}

  clval &= b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '&=' is a traditional token spelling, consider using an alternative token 'and_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval and_eq b;{{$}}

  clval |= b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '|=' is a traditional token spelling, consider using an alternative token 'or_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval or_eq b;{{$}}

  value = a != b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '!=' is a traditional token spelling, consider using an alternative token 'not_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a not_eq b;{{$}}

  clval ^= a;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '^=' is a traditional token spelling, consider using an alternative token 'xor_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval xor_eq a;{{$}}
}

struct ClassO {};

ClassO& operator&=(ClassO&, const ClassO&);
ClassO& operator|=(ClassO&, const ClassO&);
ClassO& operator^=(ClassO&, const ClassO&);
bool operator!=(const ClassO&, const ClassO&);
bool operator&&(const ClassO&, const ClassO&);
bool operator||(const ClassO&, const ClassO&);
bool operator!(const ClassO&);
ClassO operator&(const ClassO&, const ClassO&);
ClassO operator|(const ClassO&, const ClassO&);
ClassO operator^(const ClassO&, const ClassO&);
ClassO operator~(const ClassO&);

void testAllTokensToAlternative(ClassO a, ClassO b) {
  int value = 0;
  ClassO clval;

  value = a||b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '||' is a traditional token spelling, consider using an alternative token 'or' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a or b;{{$}}

  value = a&&b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '&&' is a traditional token spelling, consider using an alternative token 'and' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a and b;{{$}}

  clval = a | b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '|' is a traditional token spelling, consider using an alternative token 'bitor' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a bitor b;{{$}}

  clval = a & b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '&' is a traditional token spelling, consider using an alternative token 'bitand' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a bitand b;{{$}}

  value = !a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '!' is a traditional token spelling, consider using an alternative token 'not' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = not a;{{$}}

  clval = a^b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: '^' is a traditional token spelling, consider using an alternative token 'xor' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a xor b;{{$}}

  clval = ~b;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '~' is a traditional token spelling, consider using an alternative token 'compl' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = compl b;{{$}}

  clval &= b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '&=' is a traditional token spelling, consider using an alternative token 'and_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval and_eq b;{{$}}

  clval |= b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '|=' is a traditional token spelling, consider using an alternative token 'or_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval or_eq b;{{$}}

  value = a != b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: '!=' is a traditional token spelling, consider using an alternative token 'not_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a not_eq b;{{$}}

  clval ^= a;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: '^=' is a traditional token spelling, consider using an alternative token 'xor_eq' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval xor_eq a;{{$}}
}
