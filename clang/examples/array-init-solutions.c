// Example demonstrating correct array initialization patterns
// This file compiles without errors and shows best practices

#include <stdio.h>
#include <stdlib.h>

// Example 1: Error handling with string messages
_Noreturn void handle_error(int code) {
    // ✅ CORRECT: Array of string pointers using compound literal
    const char *msg = (const char*[]) {
        [0] = "Success",
        [1] = "General error", 
        [2] = "Invalid input",
        [3] = "Memory allocation failed"
    }[code];
    
    fprintf(stderr, "Error: %s (%d)\n", msg ? msg : "Unknown", code);
    exit(code);
}

// Example 2: Status codes with different approaches
void demonstrate_array_patterns() {
    // Pattern 1: Direct string pointer array
    const char *status_messages[] = {
        [0] = "Ready",
        [1] = "Processing", 
        [2] = "Complete",
        [3] = "Error"
    };
    
    // Pattern 2: 2D character array for fixed-length strings
    const char status_codes[][12] = {
        [0] = "READY",
        [1] = "PROCESSING",
        [2] = "COMPLETE", 
        [3] = "ERROR"
    };
    
    // Pattern 3: Single character codes (when you actually want chars)
    const char single_chars[] = {
        [0] = 'R',  // Ready
        [1] = 'P',  // Processing
        [2] = 'C',  // Complete
        [3] = 'E'   // Error
    };
    
    // Usage examples
    printf("Message: %s\n", status_messages[1]);
    printf("Code: %s\n", status_codes[1]);
    printf("Char: %c\n", single_chars[1]);
}

// Example 3: Dynamic message selection
const char* get_http_status_message(int code) {
    // ✅ CORRECT: Compound literal with string pointers
    return (const char*[]) {
        [200] = "OK",
        [404] = "Not Found",
        [500] = "Internal Server Error"
    }[code];
}

// Example 4: Initialization in function parameters
void log_message(int level) {
    // ✅ CORRECT: Inline compound literal
    printf("[%s] Message logged\n", 
           (const char*[]) {
               [0] = "DEBUG",
               [1] = "INFO", 
               [2] = "WARN",
               [3] = "ERROR"
           }[level]);
}

int main() {
    demonstrate_array_patterns();
    log_message(1);
    
    const char* http_msg = get_http_status_message(404);
    if (http_msg) {
        printf("HTTP Status: %s\n", http_msg);
    }
    
    return 0;
} 