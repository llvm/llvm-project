// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


typedef struct Connection_Type {
   long    to;
   char    type[10];
   long    length;
} Connection;

Connection link[3]
= { {1, "link1", 10},
    {2, "link2", 20},
    {3, "link3", 30} };

