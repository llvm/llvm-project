//===-- mem_map.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mem_map.h"

#include "common.h"

namespace scudo {

bool MemMapDefault::mapImpl(uptr Addr, uptr Size, const char *Name,
                            uptr Flags) {
  void *MappedAddr =
      ::scudo::map(reinterpret_cast<void *>(Addr), Size, Name, Flags, &Data);
  if (MappedAddr == nullptr)
    return false;
  Base = reinterpret_cast<uptr>(MappedAddr);
  Capacity = Size;
  return true;
}

void MemMapDefault::unmapImpl(uptr Addr, uptr Size) {
  if (Size == Capacity) {
    Base = Capacity = 0;
  } else {
    if (Base == Addr)
      Base = Addr + Size;
    Capacity -= Size;
  }

  ::scudo::unmap(reinterpret_cast<void *>(Addr), Size, UNMAP_ALL, &Data);
}

bool MemMapDefault::remapImpl(uptr Addr, uptr Size, const char *Name,
                              uptr Flags) {
  void *RemappedAddr =
      ::scudo::map(reinterpret_cast<void *>(Addr), Size, Name, Flags, &Data);
  return reinterpret_cast<uptr>(RemappedAddr) == Addr;
}

void MemMapDefault::releaseAndZeroPagesToOSImpl(uptr From, uptr Size) {
  return ::scudo::releasePagesToOS(Base, From - Base, Size, &Data);
}

void ReservedMemoryDefault::releaseImpl() {
  ::scudo::unmap(reinterpret_cast<void *>(Base), Capacity, UNMAP_ALL, &Data);
}

bool ReservedMemoryDefault::createImpl(uptr Addr, uptr Size, const char *Name,
                                       uptr Flags) {
  void *Reserved = ::scudo::map(reinterpret_cast<void *>(Addr), Size, Name,
                                Flags | MAP_NOACCESS, &Data);
  if (Reserved == nullptr)
    return false;

  Base = reinterpret_cast<uptr>(Reserved);
  Capacity = Size;

  return true;
}

ReservedMemoryDefault::MemMapT ReservedMemoryDefault::dispatchImpl(uptr Addr,
                                                                   uptr Size) {
  ReservedMemoryDefault::MemMapT NewMap(Addr, Size);
  NewMap.setMapPlatformData(Data);
  return NewMap;
}

} // namespace scudo
