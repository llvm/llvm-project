// RUN: %check_clang_tidy -std=c++11 %s readability-use-cpp-style-comments %t -config="{CheckOptions: [{key: readability-use-cpp-style-comments.ExcludeDoxygenStyleComments, value: true}]}"

// Single-line full C-style comment
static const int CONSTANT = 42; /* Important constant value */
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
// CHECK-FIXES: static const int CONSTANT = 42; // Important constant value

// Inline comment that should NOT be transformed
int a = /* inline comment */ 5;

// Multiline full-line comment
/* This is a multiline comment
   that spans several lines
   and should be converted to C++ style */
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
// CHECK-FIXES: // This is a multiline comment
// CHECK-FIXES: // that spans several lines
// CHECK-FIXES: // and should be converted to C++ style
void fnWithSomeBools(bool A,bool B) {}
// Function with parameter inline comments
void processData(int data /* input data */, 
                 bool validate /* perform validation */) {
    // These inline comments should NOT be transformed
    fnWithSomeBools(/*ControlsA=*/ true, /*ControlsB=*/ false);
}

int calculateSomething() { return 1;}
// Comment at end of complex line
int complexCalculation = calculateSomething(); /* Result of complex calculation */
// CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
// CHECK-FIXES: int complexCalculation = calculateSomething(); // Result of complex calculation

// Nested comments and edge cases
void edgeCaseFunction() {
    int x = 10 /* First value */ + 20 /* Second value */;  // Inline comments should not transform
    
    /* Comment with special characters !@#$%^&*()_+ */
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
    // CHECK-FIXES: // Comment with special characters !@#$%^&*()_+
}

// Multiline comment with various indentations
    /* This comment is indented
       and should preserve indentation when converted */
// CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
// CHECK-FIXES:     // This comment is indented
// CHECK-FIXES:     // and should preserve indentation when converted

// Complex function with mixed comment types
void complexFunction() {
    /* Full line comment at start of block */
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
    // CHECK-FIXES: // Full line comment at start of block
    
    int x = 10; /* Inline comment not to be transformed */
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use C++ style comments '//' instead of C style comments '/*...*/' [readability-use-cpp-style-comments]
    // CHECK-FIXES: int x = 10; // Inline comment not to be transformed
}

// Tests for Doxygen comments with ExcludeDoxygenStyleComments enabled
/**
 * This is a Doxygen comment for a function.
 * It should NOT be transformed.
 */
void doxygenFunction1();

/*!
 * This is another Doxygen-style comment.
 * It should also NOT be transformed.
 */
void doxygenFunction2();

/**
 * Multiline Doxygen comment describing parameters.
 *
 * @param x The first parameter.
 * @param y The second parameter.
 * @return A result value.
 */
int doxygenFunctionWithParams(int x, int y);

/*******************************
 * Non-Doxygen block comments without markers
 *******************************/
void DoxygenBlock();

/*!
 * This is a single-line Doxygen comment.
 * Should NOT be transformed.
 */
void singleLineDoxygen();
