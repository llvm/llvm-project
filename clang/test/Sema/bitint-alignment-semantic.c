// RUN: %clang_cc1 -triple=i386-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -fsyntax-only -verify %s

// Test that _BitInt alignment is computed correctly during semantic analysis
// This test verifies the fix for the alignment discrepancy between Clang and GCC

#include <stdint.h>

// Test that _BitInt alignment constants are computed correctly
void test_static_asserts() {
    // Basic alignment tests
    _Static_assert(_Alignof(_BitInt(8)) == 1, "");
    _Static_assert(_Alignof(_BitInt(16)) == 2, "");
    _Static_assert(_Alignof(_BitInt(32)) == 4, "");
    
    // The key test: _BitInt(64) should have 8-byte alignment
    _Static_assert(_Alignof(_BitInt(64)) == 8, ""); // expected-warning {{comparison of unsigned expression < 0 is always false}}
    
    // Test larger sizes
    _Static_assert(_Alignof(_BitInt(128)) == 8, "");
}

// Test that __alignof also returns correct values
void test_gnu_alignof() {
    // These should all compile without warnings
    int a = __alignof(_BitInt(8));
    int b = __alignof(_BitInt(16));
    int c = __alignof(_BitInt(32));
    int d = __alignof(_BitInt(64));  // Should be 8, not 4
    
    // Verify the values are correct
    _Static_assert(__alignof(_BitInt(64)) == 8, "");
}

// Test struct alignment
struct test_struct {
    char c;
    _BitInt(64) bi;  // Should be 8-byte aligned
    long long ll;     // Should be 8-byte aligned
};

void test_struct_alignment() {
    // Verify struct alignment is computed correctly
    _Static_assert(_Alignof(struct test_struct) >= 8, "");
    
    // Test individual member alignment
    _Static_assert(_Alignof(((struct test_struct*)0)->bi) == 8, "");
    _Static_assert(_Alignof(((struct test_struct*)0)->ll) == 8, "");
}

// Test array alignment
void test_array_alignment() {
    _BitInt(64) arr[4];
    
    // Array alignment should match element alignment
    _Static_assert(_Alignof(arr) == 8, "");
    _Static_assert(__alignof(arr) == 8, "");
}

// Test that alignment is consistent between different contexts
void test_alignment_consistency() {
    _BitInt(64) var;
    
    // All these should return the same value (8)
    int a1 = _Alignof(_BitInt(64));
    int a2 = _Alignof(var);
    int a3 = __alignof(_BitInt(64));
    int a4 = __alignof(var);
    
    // Verify consistency
    _Static_assert(_Alignof(_BitInt(64)) == __alignof(_BitInt(64)), "");
    _Static_assert(_Alignof(var) == __alignof(var), "");
}

// Test that the fix doesn't break other _BitInt sizes
void test_other_sizes() {
    _Static_assert(_Alignof(_BitInt(1)) == 1, "");
    _Static_assert(_Alignof(_BitInt(7)) == 1, "");
    _Static_assert(_Alignof(_BitInt(9)) == 2, "");
    _Static_assert(_Alignof(_BitInt(15)) == 2, "");
    _Static_assert(_Alignof(_BitInt(17)) == 4, "");
    _Static_assert(_Alignof(_BitInt(31)) == 4, "");
    _Static_assert(_Alignof(_BitInt(33)) == 8, "");
    _Static_assert(_Alignof(_BitInt(63)) == 8, "");
    _Static_assert(_Alignof(_BitInt(65)) == 8, "");
    _Static_assert(_Alignof(_BitInt(127)) == 8, "");
    _Static_assert(_Alignof(_BitInt(129)) == 8, "");
} 