// RUN: %check_clang_tidy -check-suffix=FROM %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: false, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: true \
// RUN:     }}' -- -std=c23
// RUN: %check_clang_tidy -check-suffix=TO %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: true, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: false \
// RUN:     }}' -- -std=c23
// RUN: %check_clang_tidy -check-suffix=NORMAL %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: false, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: false \
// RUN:     }}' -- -std=c23
// RUN: %check_clang_tidy -check-suffix=TO,FROM %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: true, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: true \
// RUN:     }}' -- -std=c23

// Test various implicit bool conversions in different contexts
void TestImplicitBoolConversion() {
    // Basic type conversions to bool
    int intValue = 42;
    if (intValue) // CHECK-MESSAGES-TO: :[[@LINE]]:9: warning: implicit conversion 'int' -> 'bool' [readability-implicit-bool-conversion]
                  // CHECK-FIXES-TO: if (intValue != 0)
        (void)0;

    float floatValue = 3.14f;
    while (floatValue) // CHECK-MESSAGES-TO: :[[@LINE]]:12: warning: implicit conversion 'float' -> 'bool' [readability-implicit-bool-conversion]
                       // CHECK-FIXES-TO: while (floatValue != 0.0f)
        break;

    char charValue = 'a';
    do {
        break;
    } while (charValue); // CHECK-MESSAGES-TO: :[[@LINE]]:14: warning: implicit conversion 'char' -> 'bool' [readability-implicit-bool-conversion]
                         // CHECK-FIXES-TO: } while (charValue != 0);

    // Pointer conversions to bool
    int* ptrValue = &intValue;
    if (ptrValue) // CHECK-MESSAGES-TO: :[[@LINE]]:9: warning: implicit conversion 'int *' -> 'bool' [readability-implicit-bool-conversion]
                  // CHECK-FIXES-TO: if (ptrValue != nullptr)
        (void)0;

    // Conversions from bool to other types
    bool boolValue = true;
    int intFromBool = boolValue; // CHECK-MESSAGES-FROM: :[[@LINE]]:23: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
                                 // CHECK-FIXES-FROM: int intFromBool = static_cast<int>(boolValue);
                                 
    float floatFromBool = boolValue; // CHECK-MESSAGES-FROM: :[[@LINE]]:27: warning: implicit conversion 'bool' -> 'float' [readability-implicit-bool-conversion]
                                     // CHECK-FIXES-FROM: float floatFromBool = static_cast<float>(boolValue);

    char charFromBool = boolValue; // CHECK-MESSAGES-FROM: :[[@LINE]]:25: warning: implicit conversion 'bool' -> 'char' [readability-implicit-bool-conversion]
                                   // CHECK-FIXES-FROM: char charFromBool = static_cast<char>(boolValue);
}
