// expected-warning@+2{{macro 'Foo' has been marked as deprecated}}
// expected-warning@+1{{macro 'Foo' has been marked as unsafe for use in headers}}
#if Foo
#endif
