// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test for "incompatible pointer to integer conversion" error when initializing 
// const char arrays with string literals in designated initializers

void test_array_initialization_errors() {
    // ❌ ERROR: Trying to initialize array of single chars with string literals
    const char *msg1 = (const char[]) {
        [0] = "Success",     // expected-error {{incompatible pointer to integer conversion initializing 'const char' with an expression of type 'char[8]'}}
        [1] = "Failed"       // expected-error {{incompatible pointer to integer conversion initializing 'const char' with an expression of type 'char[7]'}}
    }[0]; // expected-error {{incompatible integer to pointer conversion initializing 'const char *' with an expression of type 'const char'}}
    
    // ❌ ERROR: Same issue with regular array initialization
    const char error_array[] = {
        [0] = "Success",     // expected-error {{incompatible pointer to integer conversion initializing 'const char' with an expression of type 'char[8]'}}
        [1] = "Failed"       // expected-error {{incompatible pointer to integer conversion initializing 'const char' with an expression of type 'char[7]'}}
    };
}

void test_correct_solutions() {
    // ✅ CORRECT: Array of string pointers
    const char *msg1 = (const char*[]) {
        [0] = "Success",
        [1] = "Failed", 
        [2] = "Pending"
    }[0]; // expected-no-diagnostics
    
    // ✅ CORRECT: Array of single characters
    const char char_array[] = {
        [0] = 'S',  // Single character literals
        [1] = 'F',
        [2] = 'P'
    }; // expected-no-diagnostics
    
    // ✅ CORRECT: 2D character array (array of character arrays)
    const char string_array[][10] = {
        [0] = "Success",
        [1] = "Failed",
        [2] = "Pending"  
    }; // expected-no-diagnostics
    
    // ✅ CORRECT: Direct string pointer array declaration
    const char *messages[] = {
        [0] = "Success",
        [1] = "Failed",
        [2] = "Pending"
    }; // expected-no-diagnostics
} 