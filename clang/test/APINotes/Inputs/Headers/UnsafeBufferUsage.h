void unsafeFunc(int *p, int n);
[[clang::unsafe_buffer_usage]] void annotatedUnsafeFunc(int *p, int n);
[[clang::unsafe_buffer_usage]] void falseAPINotesButAnnotatedUnsafeFunc(int *p, int n);
