#include <cstdint>
#include <cstdio>

struct Simple {
  int x_;
  Simple() {
    x_ = 5;
  }
  ~Simple() {
    x_ += 1;
  }
};

Simple *volatile SimpleSink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 4) return 0;
  if (Data[0] == 'F' && Data[1] == 'U' && Data[2] == 'Z' && Data[3] == 'Z') {
    {
      Simple S;
      SimpleSink = &S;
    }
    if (SimpleSink->x_) fprintf(stderr, "Failed to catch use-after-dtor\n");
  }
  return 0;
}

