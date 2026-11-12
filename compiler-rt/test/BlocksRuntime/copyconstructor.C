//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <Block.h>

// CONFIG C++ rdar://6243400,rdar://6289367


int constructors = 0;
int destructors = 0;


#define CONST const

class TestObject
{
public:
	TestObject(CONST TestObject& inObj);
	TestObject();
	~TestObject();
	
	TestObject& operator=(CONST TestObject& inObj);

	int version() CONST { return _version; }
private:
	mutable int _version;
};

TestObject::TestObject(CONST TestObject& inObj)
	
{
        ++constructors;
        _version = inObj._version;
	//printf("%p (%d) -- TestObject(const TestObject&) called\n", this, _version); 
}


TestObject::TestObject()
{
        _version = ++constructors;
	//printf("%p (%d) -- TestObject() called\n", this, _version); 
}


TestObject::~TestObject()
{
	//printf("%p -- ~TestObject() called\n", this);
        ++destructors;
}


TestObject& TestObject::operator=(CONST TestObject& inObj)
{
	//printf("%p -- operator= called\n", this);
        _version = inObj._version;
	return *this;
}



void testRoutine() {
    TestObject one;
    
    void (^b)(void) = ^{ printf("my const copy of one is %d\n", one.version()); };
}
    
    

int main(int argc, char *argv[]) {
    testRoutine();
    if (constructors == 0) {
        printf("No copy constructors!!!\n");
        return 1;
    }
    if (constructors != destructors) {
        printf("%d constructors but only %d destructors\n", constructors, destructors);
        return 1;
    }
    printf("%s:success\n", argv[0]);
    return 0;
}
