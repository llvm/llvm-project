//===-- ConvertUTFWrapper.cpp - Wrap ConvertUTF.h with clang data types -----===
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/ConvertUTF.h"
#include "clang/Basic/LLVM.h"

namespace clang {

bool ConvertUTF8toWide(unsigned WideCharWidth, llvm::StringRef Source,
                       char *&ResultPtr, const UTF8 *&ErrorPtr) {
  assert(WideCharWidth == 1 || WideCharWidth == 2 || WideCharWidth == 4);
  ConversionResult result = conversionOK;
  // Copy the character span over.
  if (WideCharWidth == 1) {
    const UTF8 *Pos = reinterpret_cast<const UTF8*>(Source.begin());
    if (!isLegalUTF8String(&Pos, reinterpret_cast<const UTF8*>(Source.end()))) {
      result = sourceIllegal;
      ErrorPtr = Pos;
    } else {
      memcpy(ResultPtr, Source.data(), Source.size());
      ResultPtr += Source.size();
    }
  } else if (WideCharWidth == 2) {
    const UTF8 *sourceStart = (const UTF8*)Source.data();
    // FIXME: Make the type of the result buffer correct instead of
    // using reinterpret_cast.
    UTF16 *targetStart = reinterpret_cast<UTF16*>(ResultPtr);
    ConversionFlags flags = strictConversion;
    result = ConvertUTF8toUTF16(
        &sourceStart, sourceStart + Source.size(),
        &targetStart, targetStart + 2*Source.size(), flags);
    if (result == conversionOK)
      ResultPtr = reinterpret_cast<char*>(targetStart);
    else
      ErrorPtr = sourceStart;
  } else if (WideCharWidth == 4) {
    const UTF8 *sourceStart = (const UTF8*)Source.data();
    // FIXME: Make the type of the result buffer correct instead of
    // using reinterpret_cast.
    UTF32 *targetStart = reinterpret_cast<UTF32*>(ResultPtr);
    ConversionFlags flags = strictConversion;
    result = ConvertUTF8toUTF32(
        &sourceStart, sourceStart + Source.size(),
        &targetStart, targetStart + 4*Source.size(), flags);
    if (result == conversionOK)
      ResultPtr = reinterpret_cast<char*>(targetStart);
    else
      ErrorPtr = sourceStart;
  }
  assert((result != targetExhausted)
         && "ConvertUTF8toUTFXX exhausted target buffer");
  return result == conversionOK;
}

bool ConvertCodePointToUTF8(unsigned Source, char *&ResultPtr) {
  const UTF32 *SourceStart = &Source;
  const UTF32 *SourceEnd = SourceStart + 1;
  UTF8 *TargetStart = reinterpret_cast<UTF8 *>(ResultPtr);
  UTF8 *TargetEnd = TargetStart + 4;
  ConversionResult CR = ConvertUTF32toUTF8(&SourceStart, SourceEnd,
                                           &TargetStart, TargetEnd,
                                           strictConversion);
  if (CR != conversionOK)
    return false;

  ResultPtr = reinterpret_cast<char*>(TargetStart);
  return true;
}

} // end namespace clang
