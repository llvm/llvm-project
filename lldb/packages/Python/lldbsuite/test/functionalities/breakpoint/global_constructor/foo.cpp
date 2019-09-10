#include "foo.h"

Foo::Foo() : x(42) {
    bool some_code = x == 42;  // !BR_foo
} 

Foo FooObj;
