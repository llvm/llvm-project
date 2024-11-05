int *funcToAnnotate(int *p);

// TODO: support annotating ctors and 'this'.
struct MyClass {
    MyClass(int*);
    int *annotateThis();
    int *methodToAnnotate(int *p);
};
