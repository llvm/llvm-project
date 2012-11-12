// RUN: %clang_cc1 %s -emit-llvm -o -

#pragma pack(push, 2)
struct H {
  unsigned long f1;
  unsigned long f2;
  union {
    struct opaque1 *f3;
    struct opaque2 *f4;
    struct {
      struct opaque3 *f5;
      unsigned short  f6;
    } f7;
  } f8;
};
#pragma pack(pop)

struct E {
  unsigned long f1;
  unsigned long f2;
};

typedef long (*FuncPtr) ();

extern long bork(FuncPtr handler, const struct E *list);

static long hndlr()
{
  struct H cmd = { 4, 412 };
  struct H cmd2 = { 4, 412, 0 };
  return 0;
}
void foo(void *inWindow) {
  static const struct E events[] = {
    { 123124, 1 }
  };
  bork(hndlr, events);
}
