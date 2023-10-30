// RUN: %check_clang_tidy %s readability-use-explicit-namespaces %t

namespace foo
{
void doSomething()
{
}
}

void test1()
{
        foo::doSomething();
}

using namespace foo;

void test2()
{
	// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: Missing namespace qualifiers foo::
        doSomething();
}

