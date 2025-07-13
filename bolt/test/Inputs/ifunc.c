// This test checks that IFUNC trampoline is properly recognised by BOLT

static void foo() {}
static void bar() {}

extern int use_foo;

static void *resolver_foo(void) { return use_foo ? foo : bar; }

__attribute__((ifunc("resolver_foo"))) void ifoo();

void _start() { ifoo(); }
