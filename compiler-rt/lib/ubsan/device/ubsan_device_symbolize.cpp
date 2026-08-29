//===-- ubsan_device_symbolize.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ubsan_device_symbolize.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_posix.h"

using namespace __sanitizer;

namespace __ubsan {
namespace {

// The symbolizer requires a file path and offset.
struct DeviceImage {
  uptr LoadBase;
  uptr LoadSize;
  void *Bytes;
  uptr BytesSize;
  char Path[256];
};

Mutex ImageMutex;
InternalMmapVectorNoCtor<DeviceImage> Images;

DeviceImage *ImageFor(uptr PC) {
  for (uptr I = 0; I < Images.size(); ++I) {
    DeviceImage &Img = Images[I];
    if (PC >= Img.LoadBase && PC < Img.LoadBase + Img.LoadSize)
      return &Img;
  }
  return nullptr;
}

void Drop(uptr I) {
  DeviceImage &Img = Images[I];
  if (Img.Path[0])
    internal_unlink(Img.Path);
  if (Img.Bytes)
    UnmapOrDie(Img.Bytes, Img.BytesSize);
  if (I + 1 != Images.size())
    Images[I] = Images.back();
  Images.pop_back();
}

// Device images are written to disk so they can be passed to 'llvm-symbolizer'.
const char *PathFor(DeviceImage &Img) {
  if (Img.Path[0])
    return Img.Path;
  if (!Img.Bytes)
    return nullptr;

  const char *Tmp = GetEnv("TMPDIR");
  char Binary[256];
  const char *Name = "ubsan";
  if (ReadBinaryNameCached(Binary, sizeof(Binary)))
    Name = StripModuleName(Binary);
  internal_snprintf(Img.Path, sizeof(Img.Path), "%s/%s.%d.%zx.elf",
                    Tmp ? Tmp : "/tmp", Name, (int)internal_getpid(),
                    Img.LoadBase);

  fd_t Fd = OpenFile(Img.Path, WrOnly);
  bool Ok = Fd != kInvalidFd && WriteToFile(Fd, Img.Bytes, Img.BytesSize);
  if (Fd != kInvalidFd)
    CloseFile(Fd);
  if (!Ok) {
    VReport(1, "%s: could not write %s; device frames will not be symbolized\n",
            SanitizerToolName, Img.Path);
    internal_unlink(Img.Path);
    Img.Path[0] = '\0';
    return nullptr;
  }
  return Img.Path;
}

} // namespace

// Make a local copy of the device image to use for future symbolization.
void TrackDeviceImage(uptr LoadBase, uptr LoadSize, const void *Storage,
                      uptr StorageSize) {
  Lock L(&ImageMutex);
  if (ImageFor(LoadBase))
    return;
  DeviceImage Img = {};
  Img.LoadBase = LoadBase;
  Img.LoadSize = LoadSize;
  if (Storage && StorageSize) {
    Img.Bytes = MmapOrDie(StorageSize, "ubsan device image");
    internal_memcpy(Img.Bytes, Storage, StorageSize);
    Img.BytesSize = StorageSize;
  }
  Images.push_back(Img);
}

void ForgetDeviceImage(uptr LoadBase) {
  Lock L(&ImageMutex);
  for (uptr I = 0; I < Images.size(); ++I) {
    if (Images[I].LoadBase != LoadBase)
      continue;
    Drop(I);
    return;
  }
}

void ForgetDeviceImages() {
  Lock L(&ImageMutex);
  while (Images.size())
    Drop(0);
}

// The custom symbolizer function we use for the device image, returns nullptr
// if the given program counter is not owned by any known device image.
SymbolizedStack *SymbolizeDevicePc(uptr PC) {
  if (!PC)
    return nullptr;

  // Get a file path for the image containing the program counter so we can pass
  // it to the LLVM symbolizer interface.
  char Path[256];
  uptr Offset = 0;
  {
    Lock L(&ImageMutex);
    DeviceImage *Img = ImageFor(PC);
    if (!Img)
      return nullptr;
    const char *P = PathFor(*Img);
    if (!P)
      return SymbolizedStack::New(PC);
    internal_strlcpy(Path, P, sizeof(Path));
    Offset = PC - Img->LoadBase;
  }

  SymbolizedStack *Frames = Symbolizer::GetOrInit()->SymbolizeModuleOffset(
      Path, Offset ? Offset - 1 : Offset);
  for (SymbolizedStack *F = Frames; F; F = F->next)
    F->info.address = PC;
  return Frames;
}

} // namespace __ubsan
