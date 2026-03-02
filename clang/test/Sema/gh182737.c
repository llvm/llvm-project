// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify %s

// expected-warning@+3 {{empty paragraph passed to '@param' command}}
// expected-warning@+2 {{'@param' command used in a comment that is not attached to a function declaration}}
/**
 * @param a
 */
typedef int my_int;

/**
 * @brief A callback
 *
 * @param[in] a param1
 * @return
 *      - true: ok
 *      - false: failure
 */
typedef bool (*func_t)(uint8_t a);
// expected-error@-1 {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
// expected-error@-2 {{unknown type name 'uint8_t'}}
// expected-error@-3 {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
// expected-error@-4 {{function cannot return function type 'int (int)'}}
