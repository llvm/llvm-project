# Common Array Initialization Errors in C

## Incompatible Pointer to Integer Conversion Error

### Problem Description

A common error when working with array initialization in C is:
```
error: incompatible pointer to integer conversion initializing 'const char' 
with an expression of type 'char[N]' [-Wint-conversion]
```

This error occurs when you try to initialize an array of single characters (`const char[]`) with string literals.

### Common Mistake

```c
// ❌ WRONG - This causes the error
const char *msg = (const char[]) {
    [0] = "Success",    // ERROR! String literal to single char
    [1] = "Failed"      // ERROR! String literal to single char  
}[0];
```

### The Problem

- `const char[]` declares an array of single characters
- `"Success"` is a string literal (array of characters)
- You cannot assign an array to a single character element

### Correct Solutions

#### Solution 1: Array of String Pointers
```c
// ✅ CORRECT - Array of pointers to strings
const char *msg = (const char*[]) {
    [0] = "Success",
    [1] = "Failed",
    [2] = "Pending"
}[code];
```

#### Solution 2: Array of Single Characters  
```c
// ✅ CORRECT - If you actually want single characters
const char chars[] = {
    [0] = 'S',  // Single character literals
    [1] = 'F',
    [2] = 'P'
};
```

#### Solution 3: 2D Character Array
```c  
// ✅ CORRECT - Array of character arrays
const char strings[][10] = {
    [0] = "Success",
    [1] = "Failed", 
    [2] = "Pending"
};
```

#### Solution 4: Direct Declaration
```c
// ✅ CORRECT - Simple string pointer array
const char *messages[] = {
    [0] = "Success",
    [1] = "Failed",
    [2] = "Pending"
};
```

### Key Differences

| Declaration | Type | Usage |
|-------------|------|-------|
| `const char[]` | Array of single chars | For individual characters |
| `const char*[]` | Array of string pointers | For string messages |
| `const char[][N]` | 2D char array | For fixed-length strings |

### Real-World Example Fix

```c
// Before (causes error):
_Noreturn void handle_error(int code) {
    const char *msg = (const char[]) {  // ❌ Wrong type
        [0] = "Success",
        [1] = "General error"
    }[code];
}

// After (fixed):
_Noreturn void handle_error(int code) {
    const char *msg = (const char*[]) {  // ✅ Added asterisk  
        [0] = "Success",
        [1] = "General error"
    }[code];
}
```

The fix is simple: change `const char[]` to `const char*[]` to indicate an array of string pointers instead of an array of single characters. 