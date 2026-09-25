//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Snapshot loaded device code objects so llvm-symbolizer can name GPU PCs.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_file.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_offload.h"
#include "sanitizer_posix.h"

namespace __sanitizer {
namespace {

struct DeviceImage {
  uptr LoadBase;
  uptr LoadSize;
  void* Bytes;
  uptr BytesSize;
  char Path[256];
};

Mutex ImageMutex;
InternalMmapVectorNoCtor<DeviceImage> Images;

DeviceImage* ImageFor(uptr PC) {
  for (uptr I = 0; I < Images.size(); ++I) {
    DeviceImage& Img = Images[I];
    if (PC >= Img.LoadBase && PC < Img.LoadBase + Img.LoadSize)
      return &Img;
  }
  return nullptr;
}

void Drop(uptr I) {
  DeviceImage& Img = Images[I];
  if (Img.Path[0])
    internal_unlink(Img.Path);
  if (Img.Bytes)
    UnmapOrDie(Img.Bytes, Img.BytesSize);
  if (I + 1 != Images.size())
    Images[I] = Images.back();
  Images.pop_back();
}

const char* PathFor(DeviceImage& Img) {
  if (Img.Path[0])
    return Img.Path;
  if (!Img.Bytes)
    return nullptr;

  const char* Tmp = GetEnv("TMPDIR");
  char Binary[256];
  const char* Name = SanitizerToolName ? SanitizerToolName : "sanitizer";
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

// True if Addr is in a tracked image. Path is null when the ELF could not
// be written.
bool SnapshotImage(uptr Addr, char** Path, uptr* Offset) {
  *Path = nullptr;
  Lock L(&ImageMutex);
  DeviceImage* Img = ImageFor(Addr);
  if (!Img)
    return false;
  *Offset = Addr - Img->LoadBase;
  if (const char* P = PathFor(*Img))
    *Path = internal_strdup(P);
  return true;
}

}  // namespace

void Offload::TrackImage(uptr LoadBase, uptr LoadSize, const void* Storage,
                         uptr StorageSize) {
  Lock L(&ImageMutex);
  if (ImageFor(LoadBase))
    return;
  DeviceImage Img = {};
  Img.LoadBase = LoadBase;
  Img.LoadSize = LoadSize;
  if (Storage && StorageSize) {
    Img.Bytes = MmapOrDie(StorageSize, "offload device image");
    internal_memcpy(Img.Bytes, Storage, StorageSize);
    Img.BytesSize = StorageSize;
  }
  Images.push_back(Img);
}

void Offload::UntrackImage(uptr LoadBase) {
  Lock L(&ImageMutex);
  for (uptr I = 0; I < Images.size(); ++I) {
    if (Images[I].LoadBase != LoadBase)
      continue;
    Drop(I);
    return;
  }
}

void Offload::UntrackImages() {
  Lock L(&ImageMutex);
  while (Images.size()) Drop(0);
}

SymbolizedStack* Offload::Symbolize(uptr PC) {
  if (!PC)
    return nullptr;

  char* Path = nullptr;
  uptr Offset = 0;
  if (!SnapshotImage(PC, &Path, &Offset))
    return nullptr;
  if (!Path)
    return SymbolizedStack::New(PC);

  SymbolizedStack* Frames = Symbolizer::GetOrInit()->SymbolizeModuleOffset(
      Path, Offset ? Offset - 1 : Offset);
  InternalFree(Path);
  for (SymbolizedStack* F = Frames; F; F = F->next) F->info.address = PC;
  return Frames;
}

}  // namespace __sanitizer
