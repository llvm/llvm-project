// RUN: %clang_cc1 %s -emit-llvm -o -

void regnode(int op);

void regnode(op)
char op;
{
}
