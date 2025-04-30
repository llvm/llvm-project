/* Test module for bug-dlsym1.c test case.  */

extern int dlopen_test_variable;

extern char foo (void);

/* here to get the unresolved symbol in our .so */
char foo(void)
{
    return dlopen_test_variable;
}
