//===- Mutagen.h - Interface header for the mutagen -------------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Define the interface between libMutagen and its consumers.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_MUTAGEN_H
#define LLVM_FUZZER_MUTAGEN_H

#include "FuzzerPlatform.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define MAX_WORD_SIZE 64

typedef struct {
  // PRNG seed.
  unsigned int Seed;

  // If non-zero, use CMP traces to guide mutations. Ignored if any of
  // |FromTORC4|, |FromTORC8|, or |FromTORCW| are null.
  int UseCmp;
  void (*FromTORC4)(size_t Idx, uint32_t *Arg1, uint32_t *Arg2);
  void (*FromTORC8)(size_t Idx, uint64_t *Arg1, uint64_t *Arg2);
  void (*FromTORCW)(size_t Idx, const uint8_t **Data1, size_t *Size1,
                    const uint8_t **Data2, size_t *Size2);

  // If non-zero, use hints from intercepting memmem, strstr, etc. Ignored if
  // |UseCmp| is zero or if |FromMMT| is null.
  int UseMemmem;
  void (*FromMMT)(size_t Idx, const uint8_t **Data, size_t *Size);

  // If non-zero, generate only ASCII (isprint+isspace) inputs.
  int OnlyASCII;

  // Optional user-provided custom mutator.
  size_t (*CustomMutator)(uint8_t *Data, size_t Size, size_t MaxSize,
                          unsigned int Seed);

  // Optional user-provided custom cross-over function.
  size_t (*CustomCrossOver)(const uint8_t *Data1, size_t Size1,
                            const uint8_t *Data2, size_t Size2, uint8_t *Out,
                            size_t MaxOutSize, unsigned int Seed);

  // Optional MemorySanitizer callbacks.
  void (*MSanUnpoison)(const volatile void *, size_t size);
  void (*MSanUnpoisonParam)(size_t n);
} LLVMMutagenConfiguration;

// Re-seeds the PRNG and sets mutator-related options.
ATTRIBUTE_INTERFACE void
LLVMMutagenConfigure(const LLVMMutagenConfiguration *config);

// Writes the mutation sequence to |Out|, and returns the number of
// characters it wrote, or would have written given a large enough buffer,
// excluding the null terminator. Thus, a return value of |Max| or greater
// indicates the sequence was truncated (like snprintf). May truncate the
// sequence unless |Verbose| is non-zero. Sets |OutNumItems| to the number of
// items in the untruncated sequence.
ATTRIBUTE_INTERFACE size_t LLVMMutagenGetMutationSequence(int Verbose,
                                                          char *Out, size_t Max,
                                                          size_t *OutNumItems);

// Writes the dictionary entry sequence to |Out|, and returns the number of
// characters it wrote, or would have written given a large enough buffer,
// excluding a null terminator. Thus, a return value of |Max| or greater
// indicates the sequence was truncated (like snprintf). May truncate the
// sequence unless |Verbose| is non-zero. Sets |OutNumItems| to the number of
// items in the untruncated sequence.
ATTRIBUTE_INTERFACE size_t LLVMMutagenGetDictionaryEntrySequence(
    int Verbose, char *Out, size_t Max, size_t *OutNumItems);

// Instructs the library to record the current mutation sequence as successful
// at increasing coverage.
ATTRIBUTE_INTERFACE void LLVMMutagenRecordSequence();

// Clears the mutation and dictionary entry sequences.
ATTRIBUTE_INTERFACE void LLVMMutagenResetSequence();

// Adds data used by various mutators to produce new inputs.
ATTRIBUTE_INTERFACE void LLVMMutagenSetCrossOverWith(const uint8_t *Data,
                                                     size_t Size);
ATTRIBUTE_INTERFACE void LLVMMutagenAddWordToDictionary(const uint8_t *Word,
                                                        size_t Size);

// Mutates the contents of |Data| and returns the new size.
ATTRIBUTE_INTERFACE size_t LLVMMutagenMutate(uint8_t *Data, size_t Size,
                                             size_t Max);

// Like |LLVMMutagenMutate|, but never selects the custom mutators and is
// therefore suitable to be called from them.
ATTRIBUTE_INTERFACE size_t LLVMMutagenDefaultMutate(uint8_t *Data, size_t Size,
                                                    size_t Max);

// Creates a recommended dictionary and returns its number of entries. The
// entries can be retrieved by subsequent calls to
// |LLVMMutagenRecommendDictionaryEntry|.
ATTRIBUTE_INTERFACE size_t LLVMMutagenRecommendDictionary();

// Returns the ASCII representation of the next recommended dictionary entry,
// or null if no entries remain (or |LLVMMutagenRecommendDictionary| wasn't
// called). If non-null, the return pointer is valid until the next call to this
// method, and if provided, |OutUseCount| is set to the entry's use count.
ATTRIBUTE_INTERFACE const char *
LLVMMutagenRecommendDictionaryEntry(size_t *OutUseCount);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // LLVM_FUZZER_MUTAGEN_H
