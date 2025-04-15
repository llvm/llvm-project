int *funcToAnnotate(int *p);

struct MyClass {
    MyClass(int*);
    int *annotateThis();
    int *annotateThis2() [[clang::lifetimebound]];
    int *methodToAnnotate(int *p);
};
