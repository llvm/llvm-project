#ifndef THIRD_PARTY_LLVM_LLVM_PROJECT_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_OBJC_ASSERT_XCTESTASSERTIONS_H_
#define THIRD_PARTY_LLVM_LLVM_PROJECT_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_OBJC_ASSERT_XCTESTASSERTIONS_H_

#define _XCTPrimitiveAssertEqual(test, expression1, expressionStr1, \
                                 expression2, expressionStr2, ...)  \
  ({                                                                \
    __typeof__(expression1) expressionValue1 = (expression1);       \
    __typeof__(expression2) expressionValue2 = (expression2);       \
    if (expressionValue1 != expressionValue2) {                     \
    }                                                               \
  })

#define _XCTPrimitiveAssertEqualObjects(test, expression1, expressionStr1, \
                                        expression2, expressionStr2, ...)  \
  ({                                                                       \
    __typeof__(expression1) expressionValue1 = (expression1);              \
    __typeof__(expression2) expressionValue2 = (expression2);              \
    if (expressionValue1 != expressionValue2) {                            \
    }                                                                      \
  })

#define XCTAssertEqual(expression1, expression2, ...)                     \
  _XCTPrimitiveAssertEqual(nil, expression1, @ #expression1, expression2, \
                           @ #expression2, __VA_ARGS__)

#define XCTAssertEqualObjects(expression1, expression2, ...)        \
  _XCTPrimitiveAssertEqualObjects(nil, expression1, @ #expression1, \
                                  expression2, @ #expression2, __VA_ARGS__)

#endif // THIRD_PARTY_LLVM_LLVM_PROJECT_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_OBJC_ASSERT_XCTESTASSERTIONS_H_
