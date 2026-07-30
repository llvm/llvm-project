// RUN: %clang_cc1 %s -emit-llvm -o -
struct c__ { unsigned int type:4; };
union A { struct c__ c;  } __attribute__((aligned(8)));
struct B {
    unsigned int retainCount;
    union A objects[];
};
void foo(union A * objects, struct B *array, unsigned long k)
{  array->objects[k] = objects[k]; }
