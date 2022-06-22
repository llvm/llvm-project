// RUN: %check_clang_tidy -format-style=LLVM -check-suffix=DEFAULT %s \
// RUN:   modernize-replace-disallow-copy-and-assign-macro %t

// RUN: %check_clang_tidy -format-style=LLVM -check-suffix=DIFFERENT-NAME %s \
// RUN:  modernize-replace-disallow-copy-and-assign-macro %t \
// RUN:  -config="{CheckOptions: [ \
// RUN:   {key: modernize-replace-disallow-copy-and-assign-macro.MacroName, \
// RUN:    value: MY_MACRO_NAME}]}"

// RUN: %check_clang_tidy -format-style=LLVM -check-suffix=FINALIZE %s \
// RUN:  modernize-replace-disallow-copy-and-assign-macro %t \
// RUN:  -config="{CheckOptions: [ \
// RUN:   {key: modernize-replace-disallow-copy-and-assign-macro.MacroName, \
// RUN:    value: DISALLOW_COPY_AND_ASSIGN_FINALIZE}]}"

// RUN: clang-tidy %s -checks="-*,modernize-replace-disallow-copy-and-assign-macro" \
// RUN:   -config="{CheckOptions: [ \
// RUN:    {key: modernize-replace-disallow-copy-and-assign-macro.MacroName, \
// RUN:     value: DISALLOW_COPY_AND_ASSIGN_MORE_AGUMENTS}]}" -- -Wno-extra-semi | count 0

// RUN: clang-tidy %s -checks="-*,modernize-replace-disallow-copy-and-assign-macro" \
// RUN:   -config="{CheckOptions: [ \
// RUN:    {key: modernize-replace-disallow-copy-and-assign-macro.MacroName, \
// RUN:     value: DISALLOW_COPY_AND_ASSIGN_NEEDS_PREEXPANSION}]}" -- -Wno-extra-semi | count 0

// Note: the last two tests expect no diagnostics, but FileCheck cannot handle
// that, hence the use of | count 0.

#define DISALLOW_COPY_AND_ASSIGN(TypeName)

class TestClass1 {
private:
  DISALLOW_COPY_AND_ASSIGN(TestClass1);
};
// CHECK-MESSAGES-DEFAULT: :[[@LINE-2]]:3: warning: prefer deleting copy constructor and assignment operator over using macro 'DISALLOW_COPY_AND_ASSIGN' [modernize-replace-disallow-copy-and-assign-macro]
// CHECK-FIXES-DEFAULT:      {{^}}  TestClass1(const TestClass1 &) = delete;{{$}}
// CHECK-FIXES-DEFAULT-NEXT: {{^}}  const TestClass1 &operator=(const TestClass1 &) = delete;{{$}}

#define MY_MACRO_NAME(TypeName)

class TestClass2 {
private:
  MY_MACRO_NAME(TestClass2);
};
// CHECK-MESSAGES-DIFFERENT-NAME: :[[@LINE-2]]:3: warning: prefer deleting copy constructor and assignment operator over using macro 'MY_MACRO_NAME' [modernize-replace-disallow-copy-and-assign-macro]
// CHECK-FIXES-DIFFERENT-NAME:      {{^}}  TestClass2(const TestClass2 &) = delete;{{$}}
// CHECK-FIXES-DIFFERENT-NAME-NEXT: {{^}}  const TestClass2 &operator=(const TestClass2 &) = delete;{{$}}

#define DISALLOW_COPY_AND_ASSIGN_FINALIZE(TypeName) \
  TypeName(const TypeName &) = delete;              \
  const TypeName &operator=(const TypeName &) = delete;

class TestClass3 {
private:
  // Notice, that the macro allows to be used without a semicolon because the
  // macro definition already contains one above. Therefore our replacement must
  // contain a semicolon at the end.
  DISALLOW_COPY_AND_ASSIGN_FINALIZE(TestClass3)
};
// CHECK-MESSAGES-FINALIZE: :[[@LINE-2]]:3: warning: prefer deleting copy constructor and assignment operator over using macro 'DISALLOW_COPY_AND_ASSIGN_FINALIZE' [modernize-replace-disallow-copy-and-assign-macro]
// CHECK-FIXES-FINALIZE:      {{^}}  TestClass3(const TestClass3 &) = delete;{{$}}
// CHECK-FIXES-FINALIZE-NEXT: {{^}}  const TestClass3 &operator=(const TestClass3 &) = delete;{{$}}

#define DISALLOW_COPY_AND_ASSIGN_MORE_AGUMENTS(A, B)

class TestClass4 {
private:
  DISALLOW_COPY_AND_ASSIGN_MORE_AGUMENTS(TestClass4, TestClass4);
};
// CHECK-MESSAGES-MORE-ARGUMENTS-NOT: warning: prefer deleting copy constructor and assignment operator over using macro 'DISALLOW_COPY_AND_ASSIGN_MORE_AGUMENTS'

#define DISALLOW_COPY_AND_ASSIGN_NEEDS_PREEXPANSION(A)
#define TESTCLASS TestClass5

class TestClass5 {
private:
  DISALLOW_COPY_AND_ASSIGN_NEEDS_PREEXPANSION(TESTCLASS);
};
// CHECK-MESSAGES-MORE-ARGUMENTS-NOT: warning: prefer deleting copy constructor and assignment operator over using macro 'DISALLOW_COPY_AND_ASSIGN_NEEDS_PREEXPANSION'
