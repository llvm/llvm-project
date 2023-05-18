// RUN: %clang_cc1 -emit-llvm %s -o -

struct QChar {unsigned short X; QChar(unsigned short); } ;

struct Command {
        Command(QChar c) : c(c) {}
        unsigned int type : 4;
        QChar c;
    };

Command X(QChar('c'));

void Foo(QChar );
void bar() { Foo(X.c); }
