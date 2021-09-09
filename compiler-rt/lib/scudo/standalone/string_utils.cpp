//===-- string_utils.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "string_utils.h"
#include "common.h"

#include <stdarg.h>
#include <string.h>

namespace scudo {

static int appendChar(char **Buffer, const char *BufferEnd, char C) {
  if (*Buffer < BufferEnd) {
    **Buffer = C;
    (*Buffer)++;
  }
  return 1;
}

// Appends number in a given Base to buffer. If its length is less than
// |MinNumberLength|, it is padded with leading zeroes or spaces, depending
// on the value of |PadWithZero|.
static int appendNumber(char **Buffer, const char *BufferEnd, u64 AbsoluteValue,
                        u8 Base, u8 MinNumberLength, bool PadWithZero,
                        bool Negative, bool Upper) {
  constexpr uptr MaxLen = 30;
  RAW_CHECK(Base == 10 || Base == 16);
  RAW_CHECK(Base == 10 || !Negative);
  RAW_CHECK(AbsoluteValue || !Negative);
  RAW_CHECK(MinNumberLength < MaxLen);
  int Res = 0;
  if (Negative && MinNumberLength)
    --MinNumberLength;
  if (Negative && PadWithZero)
    Res += appendChar(Buffer, BufferEnd, '-');
  uptr NumBuffer[MaxLen];
  int Pos = 0;
  do {
    RAW_CHECK_MSG(static_cast<uptr>(Pos) < MaxLen,
                  "appendNumber buffer overflow");
    NumBuffer[Pos++] = static_cast<uptr>(AbsoluteValue % Base);
    AbsoluteValue /= Base;
  } while (AbsoluteValue > 0);
  if (Pos < MinNumberLength) {
    memset(&NumBuffer[Pos], 0,
           sizeof(NumBuffer[0]) * static_cast<uptr>(MinNumberLength - Pos));
    Pos = MinNumberLength;
  }
  RAW_CHECK(Pos > 0);
  Pos--;
  for (; Pos >= 0 && NumBuffer[Pos] == 0; Pos--) {
    char c = (PadWithZero || Pos == 0) ? '0' : ' ';
    Res += appendChar(Buffer, BufferEnd, c);
  }
  if (Negative && !PadWithZero)
    Res += appendChar(Buffer, BufferEnd, '-');
  for (; Pos >= 0; Pos--) {
    char Digit = static_cast<char>(NumBuffer[Pos]);
    Digit = static_cast<char>((Digit < 10) ? '0' + Digit
                                           : (Upper ? 'A' : 'a') + Digit - 10);
    Res += appendChar(Buffer, BufferEnd, Digit);
  }
  return Res;
}

static int appendUnsigned(char **Buffer, const char *BufferEnd, u64 Num,
                          u8 Base, u8 MinNumberLength, bool PadWithZero,
                          bool Upper) {
  return appendNumber(Buffer, BufferEnd, Num, Base, MinNumberLength,
                      PadWithZero, /*Negative=*/false, Upper);
}

static int appendSignedDecimal(char **Buffer, const char *BufferEnd, s64 Num,
                               u8 MinNumberLength, bool PadWithZero) {
  const bool Negative = (Num < 0);
  const u64 UnsignedNum = (Num == INT64_MIN)
                              ? static_cast<u64>(INT64_MAX) + 1
                              : static_cast<u64>(Negative ? -Num : Num);
  return appendNumber(Buffer, BufferEnd, UnsignedNum, 10, MinNumberLength,
                      PadWithZero, Negative, /*Upper=*/false);
}

// Use the fact that explicitly requesting 0 Width (%0s) results in UB and
// interpret Width == 0 as "no Width requested":
// Width == 0 - no Width requested
// Width  < 0 - left-justify S within and pad it to -Width chars, if necessary
// Width  > 0 - right-justify S, not implemented yet
static int appendString(char **Buffer, const char *BufferEnd, int Width,
                        int MaxChars, const char *S) {
  if (!S)
    S = "<null>";
  int Res = 0;
  for (; *S; S++) {
    if (MaxChars >= 0 && Res >= MaxChars)
      break;
    Res += appendChar(Buffer, BufferEnd, *S);
  }
  // Only the left justified strings are supported.
  while (Width < -Res)
    Res += appendChar(Buffer, BufferEnd, ' ');
  return Res;
}

static int appendPointer(char **Buffer, const char *BufferEnd, u64 ptr_value) {
  int Res = 0;
  Res += appendString(Buffer, BufferEnd, 0, -1, "0x");
  Res += appendUnsigned(Buffer, BufferEnd, ptr_value, 16,
                        SCUDO_POINTER_FORMAT_LENGTH, /*PadWithZero=*/true,
                        /*Upper=*/false);
  return Res;
}

static int formatString(char *Buffer, uptr BufferLength, const char *Format,
                        va_list Args) {
  static const char *PrintfFormatsHelp =
      "Supported formatString formats: %([0-9]*)?(z|ll)?{d,u,x,X}; %p; "
      "%[-]([0-9]*)?(\\.\\*)?s; %c\n";
  RAW_CHECK(Format);
  RAW_CHECK(BufferLength > 0);
  const char *BufferEnd = &Buffer[BufferLength - 1];
  const char *Cur = Format;
  int Res = 0;
  for (; *Cur; Cur++) {
    if (*Cur != '%') {
      Res += appendChar(&Buffer, BufferEnd, *Cur);
      continue;
    }
    Cur++;
    const bool LeftJustified = *Cur == '-';
    if (LeftJustified)
      Cur++;
    bool HaveWidth = (*Cur >= '0' && *Cur <= '9');
    const bool PadWithZero = (*Cur == '0');
    u8 Width = 0;
    if (HaveWidth) {
      while (*Cur >= '0' && *Cur <= '9')
        Width = static_cast<u8>(Width * 10 + *Cur++ - '0');
    }
    const bool HavePrecision = (Cur[0] == '.' && Cur[1] == '*');
    int Precision = -1;
    if (HavePrecision) {
      Cur += 2;
      Precision = va_arg(Args, int);
    }
    const bool HaveZ = (*Cur == 'z');
    Cur += HaveZ;
    const bool HaveLL = !HaveZ && (Cur[0] == 'l' && Cur[1] == 'l');
    Cur += HaveLL * 2;
    s64 DVal;
    u64 UVal;
    const bool HaveLength = HaveZ || HaveLL;
    const bool HaveFlags = HaveWidth || HaveLength;
    // At the moment only %s supports precision and left-justification.
    CHECK(!((Precision >= 0 || LeftJustified) && *Cur != 's'));
    switch (*Cur) {
    case 'd': {
      DVal = HaveLL  ? va_arg(Args, s64)
             : HaveZ ? va_arg(Args, sptr)
                     : va_arg(Args, int);
      Res += appendSignedDecimal(&Buffer, BufferEnd, DVal, Width, PadWithZero);
      break;
    }
    case 'u':
    case 'x':
    case 'X': {
      UVal = HaveLL  ? va_arg(Args, u64)
             : HaveZ ? va_arg(Args, uptr)
                     : va_arg(Args, unsigned);
      const bool Upper = (*Cur == 'X');
      Res += appendUnsigned(&Buffer, BufferEnd, UVal, (*Cur == 'u') ? 10 : 16,
                            Width, PadWithZero, Upper);
      break;
    }
    case 'p': {
      RAW_CHECK_MSG(!HaveFlags, PrintfFormatsHelp);
      Res += appendPointer(&Buffer, BufferEnd, va_arg(Args, uptr));
      break;
    }
    case 's': {
      RAW_CHECK_MSG(!HaveLength, PrintfFormatsHelp);
      // Only left-justified Width is supported.
      CHECK(!HaveWidth || LeftJustified);
      Res += appendString(&Buffer, BufferEnd, LeftJustified ? -Width : Width,
                          Precision, va_arg(Args, char *));
      break;
    }
    case 'c': {
      RAW_CHECK_MSG(!HaveFlags, PrintfFormatsHelp);
      Res +=
          appendChar(&Buffer, BufferEnd, static_cast<char>(va_arg(Args, int)));
      break;
    }
    case '%': {
      RAW_CHECK_MSG(!HaveFlags, PrintfFormatsHelp);
      Res += appendChar(&Buffer, BufferEnd, '%');
      break;
    }
    default: {
      RAW_CHECK_MSG(false, PrintfFormatsHelp);
    }
    }
  }
  RAW_CHECK(Buffer <= BufferEnd);
  appendChar(&Buffer, BufferEnd + 1, '\0');
  return Res;
}

int formatString(char *Buffer, uptr BufferLength, const char *Format, ...) {
  va_list Args;
  va_start(Args, Format);
  int Res = formatString(Buffer, BufferLength, Format, Args);
  va_end(Args);
  return Res;
}

void ScopedString::append(const char *Format, va_list Args) {
  va_list ArgsCopy;
  va_copy(ArgsCopy, Args);
  // formatString doesn't currently support a null buffer or zero buffer length,
  // so in order to get the resulting formatted string length, we use a one-char
  // buffer.
  char C[1];
  const uptr AdditionalLength =
      static_cast<uptr>(formatString(C, sizeof(C), Format, Args)) + 1;
  const uptr Length = length();
  String.resize(Length + AdditionalLength);
  const uptr FormattedLength = static_cast<uptr>(formatString(
      String.data() + Length, String.size() - Length, Format, ArgsCopy));
  RAW_CHECK(data()[length()] == '\0');
  RAW_CHECK(FormattedLength + 1 == AdditionalLength);
  va_end(ArgsCopy);
}

void ScopedString::append(const char *Format, ...) {
  va_list Args;
  va_start(Args, Format);
  append(Format, Args);
  va_end(Args);
}

void Printf(const char *Format, ...) {
  va_list Args;
  va_start(Args, Format);
  ScopedString Msg;
  Msg.append(Format, Args);
  outputRaw(Msg.data());
  va_end(Args);
}

} // namespace scudo
