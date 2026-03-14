// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A fuzz target that consumes a Zlib-compressed input.
// This test verifies that we can find this bug with a custom mutator.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <zlib.h>

// The fuzz target.
// Uncompress the data, crash on input starting with "FU".
// Good luck finding this w/o a custom mutator. :)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  uint8_t Uncompressed[100];
  size_t UncompressedLen = sizeof(Uncompressed);
  if (Z_OK != uncompress(Uncompressed, &UncompressedLen, Data, Size))
    return 0;
  if (UncompressedLen < 2) return 0;
  if (Uncompressed[0] == 'F' && Uncompressed[1] == 'U')
    abort();  // Boom
  return 0;
}

#ifdef CUSTOM_MUTATOR

// Forward-declare the libFuzzer's mutator callback.
extern "C" size_t
LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);

// The custom mutator:
//   * deserialize the data (in this case, uncompress).
//     * If the data doesn't deserialize, create a properly serialized dummy.
//   * Mutate the deserialized data (in this case, just call LLVMFuzzerMutate).
//   * Serialize the mutated data (in this case, compress).
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  uint8_t Uncompressed[100];
  size_t UncompressedLen = sizeof(Uncompressed);
  size_t CompressedLen = MaxSize;
  if (Z_OK != uncompress(Uncompressed, &UncompressedLen, Data, Size)) {
    // The data didn't uncompress.
    // So, it's either a broken input and we want to ignore it,
    // or we've started fuzzing from an empty corpus and we need to supply
    // out first properly compressed input.
    uint8_t Dummy[] = {'H', 'i'};
    if (Z_OK != compress(Data, &CompressedLen, Dummy, sizeof(Dummy)))
      return 0;
    // fprintf(stderr, "Dummy: max %zd res %zd\n", MaxSize, CompressedLen);
    return CompressedLen;
  }
  UncompressedLen =
      LLVMFuzzerMutate(Uncompressed, UncompressedLen, sizeof(Uncompressed));
  if (Z_OK != compress(Data, &CompressedLen, Uncompressed, UncompressedLen))
    return 0;
  return CompressedLen;
}

#endif  // CUSTOM_MUTATOR
