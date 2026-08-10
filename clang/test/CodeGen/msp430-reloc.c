// REQUIRES: msp430-registered-target
// RUN: %clang -target msp430 -fPIC -S %s -o - | FileCheck %s

// Check the compilation does not crash as it was crashing before with "-fPIC" enabled

void *alloca(unsigned int size);

// CHECK: .globl foo
short foo(char** data, char encoding)
{
	char* encoding_addr = alloca(sizeof(char));
	*encoding_addr = encoding;

	char tmp3 = *encoding_addr;
	short conv2 = tmp3;
	short and = conv2 & 0xf;

	switch (and)
	{
	case 0 :
	case 4 :
	case 10 :
		return 1;
	case 11 :
		return 2;
	}

	return 0;
}

