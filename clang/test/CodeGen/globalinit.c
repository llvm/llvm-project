// RUN: %clang_cc1 -emit-llvm %s -o %t

int A[10] = { 1,2,3,4,5 };


extern int x[];
void foo() { x[0] = 1; }
int x[10];
void bar() { x[0] = 1; }


extern int y[];
void *g = y;

int latin_ptr2len (char *p);
int (*mb_ptr2len) (char *p) = latin_ptr2len;


char string[8] = "string";   // extend init
char string2[4] = "string";  // truncate init

char *test(int c) {
 static char buf[10];
 static char *bufptr = buf;

 return c ? buf : bufptr;
}


_Bool booltest = 0;
void booltest2() {
  static _Bool booltest3 = 4;
}

// Scalars in braces.
static int a = { 1 };

// References to enums.
enum {
  EnumA, EnumB
};

int c[] = { EnumA, EnumB };

// Binary operators
int d[] = { EnumA | EnumB };

// PR1968
static int array[];
static int array[4];

