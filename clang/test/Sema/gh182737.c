// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -ast-dump -verify %s

// expected-warning@+3 2 {{empty paragraph passed to '@param' command}}
// expected-warning@+2 2 {{'@param' command used in a comment that is not attached to a function declaration}}
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
typedef int (*func_t)(int a);
