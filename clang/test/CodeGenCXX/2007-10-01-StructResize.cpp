// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

#pragma pack(4)

struct Bork {
  unsigned int f1 : 3;
  unsigned int f2 : 30;
};

void Foo(Bork *hdr) {
  hdr->f1 = 7;
  hdr->f2 = 927;
}
