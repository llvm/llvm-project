// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

// Test only flexible array member functionality specific to C++.

union VariableSizeUnion {
  int s;
  char c[];
};

@interface LastUnionIvar {
  VariableSizeUnion flexible;
}
@end

@interface NotLastUnionIvar {
  VariableSizeUnion flexible; // expected-error {{field 'flexible' with variable sized type 'VariableSizeUnion' is not at the end of class}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end


class VariableSizeClass {
public:
  int s;
  char c[];
};

@interface LastClassIvar {
  VariableSizeClass flexible;
}
@end

@interface NotLastClassIvar {
  VariableSizeClass flexible; // expected-error {{field 'flexible' with variable sized type 'VariableSizeClass' is not at the end of class}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end
