// RUN: %clang_cc1 -emit-llvm < %s -o -
// PR2419

struct Mem {
        union {
        } u;
};

struct Mem *columnMem(void){
        static const struct Mem nullMem = { {} };
}


