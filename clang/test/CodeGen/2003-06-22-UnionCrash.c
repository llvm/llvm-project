// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

struct Blend_Map_Entry {
  union {
   float Colour[5];
   double Point_Slope[2];
  } Vals;
};

void test(struct Blend_Map_Entry* Foo)
{
}

