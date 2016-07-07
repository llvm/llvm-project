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

const u8 kIdentityCodeWithPrologue[] = {
    0x55,              // push        ebp
    0x8B, 0xEC,        // mov         ebp,esp
    0x8B, 0x45, 0x08,  // mov         eax,dword ptr [ebp + 8]
    0x5D,              // pop         ebp
    0xC3,              // ret
};

const u8 kIdentityCodeWithPushPop[] = {
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

const u8 kPatchableCode1[] = {
    0xB8, 0x4B, 0x00, 0x00, 0x00,   // mov eax,4B
    0x33, 0xC9,                     // xor ecx,ecx
    0xC3,                           // ret
};

const u8 kPatchableCode2[] = {
    0x55,                           // push ebp
    0x8B, 0xEC,                     // mov ebp,esp
    0x33, 0xC0,                     // xor eax,eax
    0x5D,                           // pop ebp
    0xC3,                           // ret
};

const u8 kPatchableCode3[] = {
    0x55,                           // push ebp
    0x8B, 0xEC,                     // mov ebp,esp
    0x6A, 0x00,                     // push 0
    0xE8, 0x3D, 0xFF, 0xFF, 0xFF,   // call <func>
};

const u8 kUnpatchableCode1[] = {
    0xC3,                           // ret
};

const u8 kUnpatchableCode2[] = {
    0x33, 0xC9,                     // xor ecx,ecx
    0xC3,                           // ret
};

const u8 kUnpatchableCode3[] = {
    0x75, 0xCC,                     // jne <label>
    0x33, 0xC9,                     // xor ecx,ecx
    0xC3,                           // ret
};

const u8 kUnpatchableCode4[] = {
    0x74, 0xCC,                     // jne <label>
    0x33, 0xC9,                     // xor ecx,ecx
    0xC3,                           // ret
};

const u8 kUnpatchableCode5[] = {
    0xEB, 0x02,                     // jmp <label>
    0x33, 0xC9,                     // xor ecx,ecx
    0xC3,                           // ret
};

const u8 kUnpatchableCode6[] = {
    0xE9, 0xCC, 0xCC, 0xCC, 0xCC,   // jmp <label>
    0x90, 0x90, 0x90, 0x90,
};

const u8 kUnpatchableCode7[] = {
    0xE8, 0xCC, 0xCC, 0xCC, 0xCC,   // call <func>
    0x90, 0x90, 0x90, 0x90,
};

#endif

// A buffer holding the dynamically generated code under test.
u8* ActiveCode;
size_t ActiveCodeLength = 4096;

template<class T>
void LoadActiveCode(const T &Code, uptr *EntryPoint) {
  if (ActiveCode == nullptr) {
    ActiveCode =
        (u8*)::VirtualAlloc(nullptr, ActiveCodeLength, MEM_COMMIT | MEM_RESERVE,
                            PAGE_EXECUTE_READWRITE);
    ASSERT_NE(ActiveCode, nullptr);
  }

  size_t Position = 0;
  *EntryPoint = (uptr)&ActiveCode[0];

  // Copy the function body.
  for (size_t i = 0; i < sizeof(T); ++i)
    ActiveCode[Position++] = Code[i];
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

template<class T>
bool TestFunctionPatching(const T &Code) {
  uptr Address;
  int x = sizeof(T);

  LoadActiveCode<T>(Code, &Address);
  uptr UnusedRealAddress = 0;
  return OverrideFunction(Address, (uptr)&InterceptorFunction,
                          &UnusedRealAddress);
}

template<class T>
void TestIdentityFunctionPatching(const T &IdentityCode) {
  uptr IdentityAddress;
  LoadActiveCode<T>(IdentityCode, &IdentityAddress);
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
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop);
}

TEST(Interception, PatchableFunction) {
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode1));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode2));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode3));

  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode2));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode7));
}
#endif

}  // namespace __interception

#endif  // SANITIZER_WINDOWS
#endif  // #if !SANITIZER_DEBUG
