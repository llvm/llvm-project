#include <iostream>

extern "C" int64_t __botw_jit_code(void *JitCode, int64_t NumArgs,
                                   void **TgtArgs, ptrdiff_t *TgtOffsets,
                                   void *DeviceArgs, int64_t NumHostArgs,
                                   void **ArgBasePtrs, void **ArgPtrs,
                                   int64_t *ArgSizes, int64_t *ArgTypes,
                                   void **ArgNames) {
  char *JitCodeC = reinterpret_cast<char *>(JitCode);
  std::cerr << "Got a jit call with " << NumArgs << " args into:\n"
            << JitCodeC << "\n";

#define p(A) std::cerr << " " << #A << ": " << A[I] << "\n"
#define h(A)                                                                   \
  std::cerr << " " << #A << std::hex << ": 0x" << A[I] << std::dec << "\n"
  for (unsigned I = 0; I < NumArgs; I++) {
    std::cerr << "Device Arg #" << I << ":\n";
    p(TgtArgs);
    p(TgtOffsets);
  }
  for (unsigned I = 0; I < NumHostArgs; I++) {
    std::cerr << "Host Arg #" << I << ":\n";
    p(ArgBasePtrs);
    p(ArgPtrs);
    p(ArgSizes);
    h(ArgTypes);
  }
#undef p
#undef h

  return 0;
}
