// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

typedef struct
{
	unsigned short a : 1;
	unsigned short b : 2;
	unsigned short c : 1;
	unsigned short d : 1;
	unsigned short e : 1;
	unsigned short f : 1;
	unsigned short g : 2;
	unsigned short : 7;
	union
	{
		struct
		{
			unsigned char h : 1;
			unsigned char i : 1;
			unsigned char j : 1;
			unsigned char : 5;
		};
		struct
		{
			unsigned char k : 3;
			unsigned char : 5;
		};
	};
	unsigned char : 8;
} tt;

typedef struct
{
 unsigned char s;
 tt t;
 unsigned int u;
} ttt;

ttt X = {
    4,
       { 0 },
	55,
};
