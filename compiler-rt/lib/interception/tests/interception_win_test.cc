//===-- interception_win_test.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for interception_win.h.
//
//===----------------------------------------------------------------------===//
#include "interception/interception.h"

#include "gtest/gtest.h"

// Too slow for debug build
#if !SANITIZER_DEBUG
#if SANITIZER_WINDOWS

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

typedef int (*IdentityFunction)(int);

#if !SANITIZER_WINDOWS64
u8 kIdentityCodeWithPrologue[] = {
    0x55,              // push        ebp
    0x8B, 0xEC,        // mov         ebp,esp
    0x8B, 0x45, 0x08,  // mov         eax,dword ptr [ebp + 8]
    0x5D,              // pop         ebp
    0xC3,              // ret
};

u8 kIdentityCodeWithPushPop[] = {
    0x55,              // push        ebp
    0x8B, 0xEC,        // mov         ebp,esp
    0x53,              // push        ebx
    0x50,              // push        eax
    0x58,              // pop         eax
    0x8B, 0x45, 0x08,  // mov         eax,dword ptr [ebp + 8]
    0x5B,              // pop         ebx
    0x5D,              // pop         ebp
    0xC3,              // ret
};

#endif

// A buffer holding the dynamically generated code under test.
u8* ActiveCode;
size_t ActiveCodeLength = 4096;

bool LoadActiveCode(u8* Code, size_t CodeLength, uptr* EntryPoint) {
  if (ActiveCode == nullptr) {
    ActiveCode =
        (u8*)::VirtualAlloc(nullptr, ActiveCodeLength, MEM_COMMIT | MEM_RESERVE,
                            PAGE_EXECUTE_READWRITE);
    if (ActiveCode == nullptr) return false;
  }

  size_t Position = 0;
  *EntryPoint = (uptr)&ActiveCode[0];

  // Copy the function body.
  for (size_t i = 0; i < CodeLength; ++i)
  	ActiveCode[Position++] = Code[i];

  return true;
}

int InterceptorFunctionCalled;

NOINLINE int InterceptorFunction(int x) {
  ++InterceptorFunctionCalled;
  return x;
}

}  // namespace

namespace __interception {

// Tests for interception_win.h
TEST(Interception, InternalGetProcAddress) {
  HMODULE ntdll_handle = ::GetModuleHandle("ntdll");
  ASSERT_NE(nullptr, ntdll_handle);
  uptr DbgPrint_expected = (uptr)::GetProcAddress(ntdll_handle, "DbgPrint");
  uptr isdigit_expected = (uptr)::GetProcAddress(ntdll_handle, "isdigit");
  uptr DbgPrint_adddress = InternalGetProcAddress(ntdll_handle, "DbgPrint");
  uptr isdigit_address = InternalGetProcAddress(ntdll_handle, "isdigit");

  EXPECT_EQ(DbgPrint_expected, DbgPrint_adddress);
  EXPECT_EQ(isdigit_expected, isdigit_address);
  EXPECT_NE(DbgPrint_adddress, isdigit_address);
}

void TestIdentityFunctionPatching(u8* IdentityCode, size_t IdentityCodeLength) {
  uptr IdentityAddress;
  ASSERT_TRUE(
      LoadActiveCode(IdentityCode, IdentityCodeLength, &IdentityAddress));
  IdentityFunction Identity = (IdentityFunction)IdentityAddress;

  // Validate behavior before dynamic patching.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(0, Identity(0));
  EXPECT_EQ(42, Identity(42));
  EXPECT_EQ(0, InterceptorFunctionCalled);

  // Patch the function.
  uptr RealIdentityAddress = 0;
  EXPECT_TRUE(OverrideFunction(IdentityAddress, (uptr)&InterceptorFunction,
                               &RealIdentityAddress));
  IdentityFunction RealIdentity = (IdentityFunction)RealIdentityAddress;

  // Calling the redirected function.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(0, Identity(0));
  EXPECT_EQ(42, Identity(42));
  EXPECT_EQ(2, InterceptorFunctionCalled);

  // Calling the real function.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(0, RealIdentity(0));
  EXPECT_EQ(42, RealIdentity(42));
  EXPECT_EQ(0, InterceptorFunctionCalled);
}

#if !SANITIZER_WINDOWS64
TEST(Interception, OverrideFunction) {
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue,
                               sizeof(kIdentityCodeWithPrologue));
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop,
                               sizeof(kIdentityCodeWithPushPop));
}
#endif

}  // namespace __interception

#endif  // SANITIZER_WINDOWS
#endif  // #if !SANITIZER_DEBUG
