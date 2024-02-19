// RUN: %check_clang_tidy %s readability-operators-representation %t -- -config="{CheckOptions: {\
// RUN: readability-operators-representation.BinaryOperators: '&&;&=;&;|;~;!;!=;||;|=;^;^=', \
// RUN: readability-operators-representation.OverloadedOperators: '&&;&=;&;|;~;!;!=;||;|=;^;^='}}" --

void testAllTokensToAlternative(int a, int b) {
  int value = 0;

  value = a or b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'or' is an alternative token spelling, consider using a traditional token '||' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a || b;{{$}}

  value = a and b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'and' is an alternative token spelling, consider using a traditional token '&&' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a && b;{{$}}

  value = a bitor b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'bitor' is an alternative token spelling, consider using a traditional token '|' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a | b;{{$}}

  value = a bitand b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'bitand' is an alternative token spelling, consider using a traditional token '&' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a & b;{{$}}

  value = not a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'not' is an alternative token spelling, consider using a traditional token '!' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = ! a;{{$}}

  value = a xor b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'xor' is an alternative token spelling, consider using a traditional token '^' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a ^ b;{{$}}

  value = compl b;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'compl' is an alternative token spelling, consider using a traditional token '~' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = ~ b;{{$}}

  value and_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'and_eq' is an alternative token spelling, consider using a traditional token '&=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value &= b;{{$}}

  value or_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'or_eq' is an alternative token spelling, consider using a traditional token '|=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value |= b;{{$}}

  value = a not_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'not_eq' is an alternative token spelling, consider using a traditional token '!=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a != b;{{$}}

  value xor_eq a;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'xor_eq' is an alternative token spelling, consider using a traditional token '^=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value ^= a;{{$}}
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

  value = a or b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'or' is an alternative token spelling, consider using a traditional token '||' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a || b;{{$}}

  value = a and b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'and' is an alternative token spelling, consider using a traditional token '&&' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a && b;{{$}}

  clval = a bitor b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'bitor' is an alternative token spelling, consider using a traditional token '|' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a | b;{{$}}

  clval = a bitand b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'bitand' is an alternative token spelling, consider using a traditional token '&' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a & b;{{$}}

  value = not a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'not' is an alternative token spelling, consider using a traditional token '!' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = ! a;{{$}}

  clval = a xor b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'xor' is an alternative token spelling, consider using a traditional token '^' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a ^ b;{{$}}

  clval = compl b;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'compl' is an alternative token spelling, consider using a traditional token '~' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = ~ b;{{$}}

  clval and_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'and_eq' is an alternative token spelling, consider using a traditional token '&=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval &= b;{{$}}

  clval or_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'or_eq' is an alternative token spelling, consider using a traditional token '|=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval |= b;{{$}}

  value = a not_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'not_eq' is an alternative token spelling, consider using a traditional token '!=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a != b;{{$}}

  clval xor_eq a;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'xor_eq' is an alternative token spelling, consider using a traditional token '^=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval ^= a;{{$}}
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

  value = a or b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'or' is an alternative token spelling, consider using a traditional token '||' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a || b;{{$}}

  value = a and b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'and' is an alternative token spelling, consider using a traditional token '&&' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a && b;{{$}}

  clval = a bitor b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'bitor' is an alternative token spelling, consider using a traditional token '|' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a | b;{{$}}

  clval = a bitand b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'bitand' is an alternative token spelling, consider using a traditional token '&' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a & b;{{$}}

  value = not a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'not' is an alternative token spelling, consider using a traditional token '!' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = ! a;{{$}}

  clval = a xor b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'xor' is an alternative token spelling, consider using a traditional token '^' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = a ^ b;{{$}}

  clval = compl b;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'compl' is an alternative token spelling, consider using a traditional token '~' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval = ~ b;{{$}}

  clval and_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'and_eq' is an alternative token spelling, consider using a traditional token '&=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval &= b;{{$}}

  clval or_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'or_eq' is an alternative token spelling, consider using a traditional token '|=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval |= b;{{$}}

  value = a not_eq b;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'not_eq' is an alternative token spelling, consider using a traditional token '!=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}value = a != b;{{$}}

  clval xor_eq a;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'xor_eq' is an alternative token spelling, consider using a traditional token '^=' for consistency [readability-operators-representation]
  // CHECK-FIXES: {{^  }}clval ^= a;{{$}}
}
