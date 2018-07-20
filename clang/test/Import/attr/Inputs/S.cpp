extern void f() __attribute__((const));

struct S {
  struct {
    int a __attribute__((packed));
  };
};

void stmt() {
#pragma unroll
  for (;;)
    ;
}
