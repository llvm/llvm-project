// RUN: %clang_cc1 %s -emit-llvm -o %t

typedef __attribute__((ext_vector_type(4)))  unsigned char uchar4;
typedef __attribute__((ext_vector_type(8))) unsigned char uchar8;

// OpenCL allows vectors to be initialized by vectors Handle bug in
// VisitInitListExpr for this case below.
void foo( uchar8 x )
{
  uchar4 val[4] = {{(uchar4){x.lo}}};
}
