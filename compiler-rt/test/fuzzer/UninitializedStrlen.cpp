#include <cstdint>
#include <cstring>

volatile size_t Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 4) return 0;
  if (Data[0] == 'F' && Data[1] == 'U' && Data[2] == 'Z' && Data[3] == 'Z') {
    char uninit[7];
    Sink = strlen(uninit);
  }
  return 0;
}

