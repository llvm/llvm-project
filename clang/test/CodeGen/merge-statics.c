// RUN: %clang_cc1 < %s -emit-llvm | grep internal | count 1

// The two decls for 'a' should merge into one llvm GlobalVariable.

struct s { int x; };
static struct s a;

struct s *ap1 = &a;

static struct s a =  {
    10
};

