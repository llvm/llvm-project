//===-- interception_win_test.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// Disabling for ARM64 since testcases are x86/x64 assembly.
#if !SANITIZER_DEBUG
#if SANITIZER_WINDOWS
#    if !SANITIZER_WINDOWS_ARM64

#      include <stdarg.h>

#      define WIN32_LEAN_AND_MEAN
#      include <windows.h>

namespace __interception {
namespace {

enum FunctionPrefixKind {
  FunctionPrefixNone,
  FunctionPrefixPadding,
  FunctionPrefixHotPatch,
  FunctionPrefixDetour,
};

typedef bool (*TestOverrideFunction)(uptr, uptr, uptr*);
typedef int (*IdentityFunction)(int);

#if SANITIZER_WINDOWS64

const u8 kIdentityCodeWithPrologue[] = {
    0x55,                   // push        rbp
    0x48, 0x89, 0xE5,       // mov         rbp,rsp
    0x8B, 0xC1,             // mov         eax,ecx
    0x5D,                   // pop         rbp
    0xC3,                   // ret
};

const u8 kIdentityCodeWithPushPop[] = {
    0x55,                   // push        rbp
    0x48, 0x89, 0xE5,       // mov         rbp,rsp
    0x53,                   // push        rbx
    0x50,                   // push        rax
    0x58,                   // pop         rax
    0x8B, 0xC1,             // mov         rax,rcx
    0x5B,                   // pop         rbx
    0x5D,                   // pop         rbp
    0xC3,                   // ret
};

const u8 kIdentityTwiceOffset = 16;
const u8 kIdentityTwice[] = {
    0x55,                   // push        rbp
    0x48, 0x89, 0xE5,       // mov         rbp,rsp
    0x8B, 0xC1,             // mov         eax,ecx
    0x5D,                   // pop         rbp
    0xC3,                   // ret
    0x90, 0x90, 0x90, 0x90,
    0x90, 0x90, 0x90, 0x90,
    0x55,                   // push        rbp
    0x48, 0x89, 0xE5,       // mov         rbp,rsp
    0x8B, 0xC1,             // mov         eax,ecx
    0x5D,                   // pop         rbp
    0xC3,                   // ret
};

const u8 kIdentityCodeWithMov[] = {
    0x89, 0xC8,             // mov         eax, ecx
    0xC3,                   // ret
};

const u8 kIdentityCodeWithJump[] = {
    0xE9, 0x04, 0x00, 0x00,
    0x00,                   // jmp + 4
    0xCC, 0xCC, 0xCC, 0xCC,
    0x89, 0xC8,             // mov         eax, ecx
    0xC3,                   // ret
};

const u8 kIdentityCodeWithJumpBackwards[] = {
    0x89, 0xC8,  // mov         eax, ecx
    0xC3,        // ret
    0xE9, 0xF8, 0xFF, 0xFF,
    0xFF,  // jmp - 8
    0xCC, 0xCC, 0xCC, 0xCC,
};
const u8 kIdentityCodeWithJumpBackwardsOffset = 3;

#    else

const u8 kIdentityCodeWithPrologue[] = {
    0x55,                   // push        ebp
    0x8B, 0xEC,             // mov         ebp,esp
    0x8B, 0x45, 0x08,       // mov         eax,dword ptr [ebp + 8]
    0x5D,                   // pop         ebp
    0xC3,                   // ret
};

const u8 kIdentityCodeWithPushPop[] = {
    0x55,                   // push        ebp
    0x8B, 0xEC,             // mov         ebp,esp
    0x53,                   // push        ebx
    0x50,                   // push        eax
    0x58,                   // pop         eax
    0x8B, 0x45, 0x08,       // mov         eax,dword ptr [ebp + 8]
    0x5B,                   // pop         ebx
    0x5D,                   // pop         ebp
    0xC3,                   // ret
};

const u8 kIdentityTwiceOffset = 8;
const u8 kIdentityTwice[] = {
    0x55,                   // push        ebp
    0x8B, 0xEC,             // mov         ebp,esp
    0x8B, 0x45, 0x08,       // mov         eax,dword ptr [ebp + 8]
    0x5D,                   // pop         ebp
    0xC3,                   // ret
    0x55,                   // push        ebp
    0x8B, 0xEC,             // mov         ebp,esp
    0x8B, 0x45, 0x08,       // mov         eax,dword ptr [ebp + 8]
    0x5D,                   // pop         ebp
    0xC3,                   // ret
};

const u8 kIdentityCodeWithMov[] = {
    0x8B, 0x44, 0x24, 0x04, // mov         eax,dword ptr [esp + 4]
    0xC3,                   // ret
};

const u8 kIdentityCodeWithJump[] = {
    0xE9, 0x04, 0x00, 0x00,
    0x00,                   // jmp + 4
    0xCC, 0xCC, 0xCC, 0xCC,
    0x8B, 0x44, 0x24, 0x04, // mov         eax,dword ptr [esp + 4]
    0xC3,                   // ret
};

const u8 kIdentityCodeWithJumpBackwards[] = {
    0x8B, 0x44, 0x24, 0x04,  // mov         eax,dword ptr [esp + 4]
    0xC3,                    // ret
    0xE9, 0xF6, 0xFF, 0xFF,
    0xFF,  // jmp - 10
    0xCC, 0xCC, 0xCC, 0xCC,
};
const u8 kIdentityCodeWithJumpBackwardsOffset = 5;

#    endif

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

const u8 kPatchableCode4[] = {
    0xE9, 0xCC, 0xCC, 0xCC, 0xCC,   // jmp <label>
    0x90, 0x90, 0x90, 0x90,
};

const u8 kPatchableCode5[] = {
    0x55,                                      // push    ebp
    0x8b, 0xec,                                // mov     ebp,esp
    0x8d, 0xa4, 0x24, 0x30, 0xfd, 0xff, 0xff,  // lea     esp,[esp-2D0h]
    0x54,                                      // push    esp
};

#if SANITIZER_WINDOWS64
u8 kLoadGlobalCode[] = {
  0x8B, 0x05, 0x00, 0x00, 0x00, 0x00, // mov    eax [rip + global]
  0xC3,                               // ret
};
#endif

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
    0xE8, 0xCC, 0xCC, 0xCC, 0xCC,   // call <func>
    0x90, 0x90, 0x90, 0x90,
};

#      if SANITIZER_WINDOWS64
const u8 kUnpatchableCode7[] = {
    0x33, 0xc0,                     // xor     eax,eax
    0x48, 0x85, 0xd2,               // test    rdx,rdx
    0x74, 0x10,                     // je      +16  (unpatchable)
};

const u8 kUnpatchableCode8[] = {
    0x48, 0x8b, 0xc1,               // mov     rax,rcx
    0x0f, 0xb7, 0x10,               // movzx   edx,word ptr [rax]
    0x48, 0x83, 0xc0, 0x02,         // add     rax,2
    0x66, 0x85, 0xd2,               // test    dx,dx
    0x75, 0xf4,                     // jne     -12  (unpatchable)
};

const u8 kUnpatchableCode9[] = {
    0x4c, 0x8b, 0xc1,               // mov     r8,rcx
    0x8a, 0x01,                     // mov     al,byte ptr [rcx]
    0x48, 0xff, 0xc1,               // inc     rcx
    0x84, 0xc0,                     // test    al,al
    0x75, 0xf7,                     // jne     -9  (unpatchable)
};

const u8 kPatchableCode6[] = {
    0x48, 0x89, 0x54, 0x24, 0xBB, // mov QWORD PTR [rsp + 0xBB], rdx
    0x33, 0xC9,                   // xor ecx,ecx
    0xC3,                         // ret
};

const u8 kPatchableCode7[] = {
    0x4c, 0x89, 0x4c, 0x24, 0xBB,  // mov QWORD PTR [rsp + 0xBB], r9
    0x33, 0xC9,                   // xor ecx,ecx
    0xC3,                         // ret
};

const u8 kPatchableCode8[] = {
    0x4c, 0x89, 0x44, 0x24, 0xBB, // mov QWORD PTR [rsp + 0xBB], r8
    0x33, 0xC9,                   // xor ecx,ecx
    0xC3,                         // ret
};

const u8 kPatchableCode9[] = {
    0x8a, 0x01,                     // al,byte ptr [rcx]
    0x45, 0x33, 0xc0,               // xor     r8d,r8d
    0x84, 0xc0,                     // test    al,al
};

const u8 kPatchableCode10[] = {
    0x45, 0x33, 0xc0,               // xor     r8d,r8d
    0x41, 0x8b, 0xc0,               // mov     eax,r8d
    0x48, 0x85, 0xd2,               // test    rdx,rdx
};

const u8 kPatchableCode11[] = {
    0x48, 0x83, 0xec, 0x38,         // sub     rsp,38h
    0x83, 0x64, 0x24, 0x28, 0x00,   // and     dword ptr [rsp+28h],0
};
#      endif

#      if !SANITIZER_WINDOWS64
const u8 kPatchableCode12[] = {
    0x55,                           // push    ebp
    0x53,                           // push    ebx
    0x57,                           // push    edi
    0x56,                           // push    esi
    0x8b, 0x6c, 0x24, 0x18,         // mov     ebp,dword ptr[esp+18h]
};

const u8 kPatchableCode13[] = {
    0x55,                           // push    ebp
    0x53,                           // push    ebx
    0x57,                           // push    edi
    0x56,                           // push    esi
    0x8b, 0x5c, 0x24, 0x14,         // mov     ebx,dword ptr[esp+14h]
};
#      endif

const u8 kPatchableCode14[] = {
    0x55,                           // push    ebp
    0x89, 0xe5,                     // mov     ebp,esp
    0x53,                           // push    ebx
    0x57,                           // push    edi
    0x56,                           // push    esi
};

const u8 kUnsupportedCode1[] = {
    0x0f, 0x0b,                     // ud2
    0x0f, 0x0b,                     // ud2
    0x0f, 0x0b,                     // ud2
    0x0f, 0x0b,                     // ud2
};

// A buffer holding the dynamically generated code under test.
u8* ActiveCode;
const size_t ActiveCodeLength = 4096;

int InterceptorFunction(int x);

/// Allocate code memory more than 2GB away from Base.
u8 *AllocateCode2GBAway(u8 *Base) {
  // Find a 64K aligned location after Base plus 2GB.
  size_t TwoGB = 0x80000000;
  size_t AllocGranularity = 0x10000;
  Base = (u8 *)((((uptr)Base + TwoGB + AllocGranularity)) & ~(AllocGranularity - 1));

  // Check if that location is free, and if not, loop over regions until we find
  // one that is.
  MEMORY_BASIC_INFORMATION mbi = {};
  while (sizeof(mbi) == VirtualQuery(Base, &mbi, sizeof(mbi))) {
    if (mbi.State & MEM_FREE) break;
    Base += mbi.RegionSize;
  }

  // Allocate one RWX page at the free location.
  return (u8 *)::VirtualAlloc(Base, ActiveCodeLength, MEM_COMMIT | MEM_RESERVE,
                              PAGE_EXECUTE_READWRITE);
}

template<class T>
static void LoadActiveCode(
    const T &code,
    uptr *entry_point,
    FunctionPrefixKind prefix_kind = FunctionPrefixNone) {
  if (ActiveCode == nullptr) {
    ActiveCode = AllocateCode2GBAway((u8*)&InterceptorFunction);
    ASSERT_NE(ActiveCode, nullptr) << "failed to allocate RWX memory 2GB away";
  }

  size_t position = 0;

  // Add padding to avoid memory violation when scanning the prefix.
  for (int i = 0; i < 16; ++i)
    ActiveCode[position++] = 0xC3;  // Instruction 'ret'.

  // Add function padding.
  size_t padding = 0;
  if (prefix_kind == FunctionPrefixPadding)
    padding = 16;
  else if (prefix_kind == FunctionPrefixDetour ||
           prefix_kind == FunctionPrefixHotPatch)
    padding = FIRST_32_SECOND_64(5, 6);
  // Insert |padding| instructions 'nop'.
  for (size_t i = 0; i < padding; ++i)
    ActiveCode[position++] = 0x90;

  // Keep track of the entry point.
  *entry_point = (uptr)&ActiveCode[position];

  // Add the detour instruction (i.e. mov edi, edi)
  if (prefix_kind == FunctionPrefixDetour) {
#if SANITIZER_WINDOWS64
    // Note that "mov edi,edi" is NOP in 32-bit only, in 64-bit it clears
    // higher bits of RDI.
    // Use 66,90H as NOP for Windows64.
    ActiveCode[position++] = 0x66;
    ActiveCode[position++] = 0x90;
#else
    // mov edi,edi.
    ActiveCode[position++] = 0x8B;
    ActiveCode[position++] = 0xFF;
#endif

  }

  // Copy the function body.
  for (size_t i = 0; i < sizeof(T); ++i)
    ActiveCode[position++] = code[i];
}

int InterceptorFunctionCalled;
IdentityFunction InterceptedRealFunction;

int InterceptorFunction(int x) {
  ++InterceptorFunctionCalled;
  return InterceptedRealFunction(x);
}

}  // namespace

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

template <class T>
static void TestIdentityFunctionPatching(
    const T &code, TestOverrideFunction override,
    FunctionPrefixKind prefix_kind = FunctionPrefixNone,
    int function_start_offset = 0) {
  uptr identity_address;
  LoadActiveCode(code, &identity_address, prefix_kind);
  identity_address += function_start_offset;
  IdentityFunction identity = (IdentityFunction)identity_address;

  // Validate behavior before dynamic patching.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(0, identity(0));
  EXPECT_EQ(42, identity(42));
  EXPECT_EQ(0, InterceptorFunctionCalled);

  // Patch the function.
  uptr real_identity_address = 0;
  bool success = override(identity_address,
                         (uptr)&InterceptorFunction,
                         &real_identity_address);
  EXPECT_TRUE(success);
  EXPECT_NE(0U, real_identity_address);
  IdentityFunction real_identity = (IdentityFunction)real_identity_address;
  InterceptedRealFunction = real_identity;

  // Don't run tests if hooking failed or the real function is not valid.
  if (!success || !real_identity_address)
    return;

  // Calling the redirected function.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(0, identity(0));
  EXPECT_EQ(42, identity(42));
  EXPECT_EQ(2, InterceptorFunctionCalled);

  // Calling the real function.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(0, real_identity(0));
  EXPECT_EQ(42, real_identity(42));
  EXPECT_EQ(0, InterceptorFunctionCalled);

  TestOnlyReleaseTrampolineRegions();
}

#    if !SANITIZER_WINDOWS64
TEST(Interception, OverrideFunctionWithDetour) {
  TestOverrideFunction override = OverrideFunctionWithDetour;
  FunctionPrefixKind prefix = FunctionPrefixDetour;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithMov, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithJump, override, prefix);
}
#endif  // !SANITIZER_WINDOWS64

TEST(Interception, OverrideFunctionWithRedirectJump) {
  TestOverrideFunction override = OverrideFunctionWithRedirectJump;
  TestIdentityFunctionPatching(kIdentityCodeWithJump, override);
  TestIdentityFunctionPatching(kIdentityCodeWithJumpBackwards, override,
                               FunctionPrefixNone,
                               kIdentityCodeWithJumpBackwardsOffset);
}

TEST(Interception, OverrideFunctionWithHotPatch) {
  TestOverrideFunction override = OverrideFunctionWithHotPatch;
  FunctionPrefixKind prefix = FunctionPrefixHotPatch;
  TestIdentityFunctionPatching(kIdentityCodeWithMov, override, prefix);
}

TEST(Interception, OverrideFunctionWithTrampoline) {
  TestOverrideFunction override = OverrideFunctionWithTrampoline;
  FunctionPrefixKind prefix = FunctionPrefixNone;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);

  prefix = FunctionPrefixPadding;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);
}

TEST(Interception, OverrideFunction) {
  TestOverrideFunction override = OverrideFunction;
  FunctionPrefixKind prefix = FunctionPrefixNone;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithJump, override, prefix);

  prefix = FunctionPrefixPadding;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithMov, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithJump, override, prefix);

  prefix = FunctionPrefixHotPatch;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithMov, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithJump, override, prefix);

  prefix = FunctionPrefixDetour;
  TestIdentityFunctionPatching(kIdentityCodeWithPrologue, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithPushPop, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithMov, override, prefix);
  TestIdentityFunctionPatching(kIdentityCodeWithJump, override, prefix);
}

template<class T>
static void TestIdentityFunctionMultiplePatching(
    const T &code,
    TestOverrideFunction override,
    FunctionPrefixKind prefix_kind = FunctionPrefixNone) {
  uptr identity_address;
  LoadActiveCode(code, &identity_address, prefix_kind);

  // Patch the function.
  uptr real_identity_address = 0;
  bool success = override(identity_address,
                          (uptr)&InterceptorFunction,
                          &real_identity_address);
  EXPECT_TRUE(success);
  EXPECT_NE(0U, real_identity_address);

  // Re-patching the function should not work.
  success = override(identity_address,
                     (uptr)&InterceptorFunction,
                     &real_identity_address);
  EXPECT_FALSE(success);

  TestOnlyReleaseTrampolineRegions();
}

TEST(Interception, OverrideFunctionMultiplePatchingIsFailing) {
#if !SANITIZER_WINDOWS64
  TestIdentityFunctionMultiplePatching(kIdentityCodeWithPrologue,
                                       OverrideFunctionWithDetour,
                                       FunctionPrefixDetour);
#endif

  TestIdentityFunctionMultiplePatching(kIdentityCodeWithMov,
                                       OverrideFunctionWithHotPatch,
                                       FunctionPrefixHotPatch);

  TestIdentityFunctionMultiplePatching(kIdentityCodeWithPushPop,
                                       OverrideFunctionWithTrampoline,
                                       FunctionPrefixPadding);
}

TEST(Interception, OverrideFunctionTwice) {
  uptr identity_address1;
  LoadActiveCode(kIdentityTwice, &identity_address1);
  uptr identity_address2 = identity_address1 + kIdentityTwiceOffset;
  IdentityFunction identity1 = (IdentityFunction)identity_address1;
  IdentityFunction identity2 = (IdentityFunction)identity_address2;

  // Patch the two functions.
  uptr real_identity_address = 0;
  EXPECT_TRUE(OverrideFunction(identity_address1,
                               (uptr)&InterceptorFunction,
                               &real_identity_address));
  EXPECT_TRUE(OverrideFunction(identity_address2,
                               (uptr)&InterceptorFunction,
                               &real_identity_address));
  IdentityFunction real_identity = (IdentityFunction)real_identity_address;
  InterceptedRealFunction = real_identity;

  // Calling the redirected function.
  InterceptorFunctionCalled = 0;
  EXPECT_EQ(42, identity1(42));
  EXPECT_EQ(42, identity2(42));
  EXPECT_EQ(2, InterceptorFunctionCalled);

  TestOnlyReleaseTrampolineRegions();
}

template<class T>
static bool TestFunctionPatching(
    const T &code,
    TestOverrideFunction override,
    FunctionPrefixKind prefix_kind = FunctionPrefixNone) {
  uptr address;
  LoadActiveCode(code, &address, prefix_kind);
  uptr unused_real_address = 0;
  bool result = override(
      address, (uptr)&InterceptorFunction, &unused_real_address);

  TestOnlyReleaseTrampolineRegions();
  return result;
}

TEST(Interception, PatchableFunction) {
  TestOverrideFunction override = OverrideFunction;
  // Test without function padding.
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode1, override));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode2, override));
#if SANITIZER_WINDOWS64
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode3, override));
#else
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode3, override));
#endif
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode4, override));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode5, override));
#if SANITIZER_WINDOWS64
  EXPECT_TRUE(TestFunctionPatching(kLoadGlobalCode, override));
#endif

  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode2, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6, override));
}

#if !SANITIZER_WINDOWS64
TEST(Interception, PatchableFunctionWithDetour) {
  TestOverrideFunction override = OverrideFunctionWithDetour;
  // Without the prefix, no function can be detoured.
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode1, override));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode2, override));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode3, override));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode4, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode2, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6, override));

  // With the prefix, all functions can be detoured.
  FunctionPrefixKind prefix = FunctionPrefixDetour;
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode1, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode2, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode3, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode4, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode1, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode2, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode3, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode4, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode5, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode6, override, prefix));
}
#endif  // !SANITIZER_WINDOWS64

TEST(Interception, PatchableFunctionWithRedirectJump) {
  TestOverrideFunction override = OverrideFunctionWithRedirectJump;
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode1, override));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode2, override));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode3, override));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode4, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode2, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5, override));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6, override));
}

TEST(Interception, PatchableFunctionWithHotPatch) {
  TestOverrideFunction override = OverrideFunctionWithHotPatch;
  FunctionPrefixKind prefix = FunctionPrefixHotPatch;

  EXPECT_TRUE(TestFunctionPatching(kPatchableCode1, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode2, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode3, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode4, override, prefix));
#if SANITIZER_WINDOWS64
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode6, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode7, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode8, override, prefix));
#endif
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode2, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6, override, prefix));
}

TEST(Interception, PatchableFunctionWithTrampoline) {
  TestOverrideFunction override = OverrideFunctionWithTrampoline;
  FunctionPrefixKind prefix = FunctionPrefixPadding;

  EXPECT_TRUE(TestFunctionPatching(kPatchableCode1, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode2, override, prefix));
#if SANITIZER_WINDOWS64
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode3, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode9, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode10, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode11, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode7, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode8, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode9, override, prefix));
#else
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode3, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode12, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode13, override, prefix));
#endif
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode4, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode14, override, prefix));

  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode2, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6, override, prefix));
}

TEST(Interception, UnsupportedInstructionWithTrampoline) {
  TestOverrideFunction override = OverrideFunctionWithTrampoline;
  FunctionPrefixKind prefix = FunctionPrefixPadding;

  static bool reportCalled;
  reportCalled = false;

  struct Local {
    static void Report(const char *format, ...) {
      if (reportCalled)
        FAIL() << "Report called more times than expected";
      reportCalled = true;
      ASSERT_STREQ(
          "interception_win: unhandled instruction at %p: %02x %02x %02x %02x "
          "%02x %02x %02x %02x\n",
          format);
      va_list args;
      va_start(args, format);
      u8 *ptr = va_arg(args, u8 *);
      for (int i = 0; i < 8; i++) EXPECT_EQ(kUnsupportedCode1[i], ptr[i]);
      int bytes[8];
      for (int i = 0; i < 8; i++) {
        bytes[i] = va_arg(args, int);
        EXPECT_EQ(kUnsupportedCode1[i], bytes[i]);
      }
      va_end(args);
    }
  };

  SetErrorReportCallback(Local::Report);
  EXPECT_FALSE(TestFunctionPatching(kUnsupportedCode1, override, prefix));
  SetErrorReportCallback(nullptr);

  if (!reportCalled)
    ADD_FAILURE() << "Report not called";
}

TEST(Interception, PatchableFunctionPadding) {
  TestOverrideFunction override = OverrideFunction;
  FunctionPrefixKind prefix = FunctionPrefixPadding;

  EXPECT_TRUE(TestFunctionPatching(kPatchableCode1, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode2, override, prefix));
#if SANITIZER_WINDOWS64
  EXPECT_FALSE(TestFunctionPatching(kPatchableCode3, override, prefix));
#else
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode3, override, prefix));
#endif
  EXPECT_TRUE(TestFunctionPatching(kPatchableCode4, override, prefix));

  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode1, override, prefix));
  EXPECT_TRUE(TestFunctionPatching(kUnpatchableCode2, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode3, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode4, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode5, override, prefix));
  EXPECT_FALSE(TestFunctionPatching(kUnpatchableCode6, override, prefix));
}

TEST(Interception, EmptyExportTable) {
  // We try to get a pointer to a function from an executable that doesn't
  // export any symbol (empty export table).
  uptr FunPtr = InternalGetProcAddress((void *)GetModuleHandleA(0), "example");
  EXPECT_EQ(0U, FunPtr);
}

const struct InstructionSizeData {
  size_t size;  // hold instruction size or 0 for failure,
                // e.g. on control instructions
  u8 instr[16];
  size_t rel_offset;  // filled just for instructions with two operands
                      // and displacement length of four bytes.
  const char *comment;
} data[] = {
    // clang-format off
    // sorted list
    { 0, {0x70, 0x71}, 0, "70 XX : jo XX (short conditional jump)"},
    { 0, {0x71, 0x71}, 0, "71 XX : jno XX (short conditional jump)"},
    { 0, {0x72, 0x71}, 0, "72 XX : jb XX (short conditional jump)"},
    { 0, {0x73, 0x71}, 0, "73 XX : jae XX (short conditional jump)"},
    { 0, {0x74, 0x71}, 0, "74 XX : je XX (short conditional jump)"},
    { 0, {0x75, 0x71}, 0, "75 XX : jne XX (short conditional jump)"},
    { 0, {0x76, 0x71}, 0, "76 XX : jbe XX (short conditional jump)"},
    { 0, {0x77, 0x71}, 0, "77 XX : ja XX (short conditional jump)"},
    { 0, {0x78, 0x71}, 0, "78 XX : js XX (short conditional jump)"},
    { 0, {0x79, 0x71}, 0, "79 XX : jns XX (short conditional jump)"},
    { 0, {0x7A, 0x71}, 0, "7A XX : jp XX (short conditional jump)"},
    { 0, {0x7B, 0x71}, 0, "7B XX : jnp XX (short conditional jump)"},
    { 0, {0x7C, 0x71}, 0, "7C XX : jl XX (short conditional jump)"},
    { 0, {0x7D, 0x71}, 0, "7D XX : jge XX (short conditional jump)"},
    { 0, {0x7E, 0x71}, 0, "7E XX : jle XX (short conditional jump)"},
    { 0, {0x7F, 0x71}, 0, "7F XX : jg XX (short conditional jump)"},
    { 0, {0xE8, 0x71, 0x72, 0x73, 0x74}, 0, "E8 XX XX XX XX : call <func>"},
    { 0, {0xE9, 0x71, 0x72, 0x73, 0x74}, 0, "E9 XX XX XX XX : jmp <label>"},
    { 0, {0xEB, 0x71}, 0, "EB XX : jmp XX (short jump)"},
    { 0, {0xFF, 0x25, 0x72, 0x73, 0x74, 0x75}, 0, "FF 25 XX YY ZZ WW : jmp dword ptr ds:[WWZZYYXX]"},
    { 1, {0x50}, 0, "50 : push eax / rax"},
    { 1, {0x51}, 0, "51 : push ecx / rcx"},
    { 1, {0x52}, 0, "52 : push edx / rdx"},
    { 1, {0x53}, 0, "53 : push ebx / rbx"},
    { 1, {0x54}, 0, "54 : push esp / rsp"},
    { 1, {0x55}, 0, "55 : push ebp / rbp"},
    { 1, {0x56}, 0, "56 : push esi / rsi"},
    { 1, {0x57}, 0, "57 : push edi / rdi"},
    { 1, {0x5D}, 0, "5D : pop ebp / rbp"},
    { 1, {0x90}, 0, "90 : nop"},
    { 1, {0xC3}, 0, "C3 : ret   (for small/empty function interception"},
    { 1, {0xCC}, 0, "CC : int 3  i.e. registering weak functions)"},
    { 2, {0x33, 0xC0}, 0, "33 C0 : xor eax, eax"},
    { 2, {0x33, 0xC9}, 0, "33 C9 : xor ecx, ecx"},
    { 2, {0x33, 0xD2}, 0, "33 D2 : xor edx, edx"},
    { 2, {0x6A, 0x71}, 0, "6A XX : push XX"},
    { 2, {0x84, 0xC9}, 0, "84 C9 : test cl,cl"},
    { 2, {0x84, 0xD2}, 0, "84 D2 : test dl,dl"},
    { 2, {0x84, 0xDB}, 0, "84 DB : test bl,bl"},
    { 2, {0x89, 0xc8}, 0, "89 C8 : mov eax, ecx"},
    { 2, {0x89, 0xE5}, 0, "89 E5 : mov ebp, esp"},
    { 2, {0x8A, 0x01}, 0, "8A 01 : mov al, byte ptr [ecx]"},
    { 2, {0x8B, 0xC1}, 0, "8B C1 : mov eax, ecx"},
    { 2, {0x8B, 0xEC}, 0, "8B EC : mov ebp, esp"},
    { 2, {0x8B, 0xFF}, 0, "8B FF : mov edi, edi"},
    { 3, {0x83, 0xE4, 0x72}, 0, "83 E4 XX : and esp, XX"},
    { 3, {0x83, 0xEC, 0x72}, 0, "83 EC XX : sub esp, XX"},
    { 3, {0xc2, 0x71, 0x72}, 0, "C2 XX XX : ret XX (needed for registering weak functions)"},
    { 5, {0x68, 0x71, 0x72, 0x73, 0x74}, 0, "68 XX XX XX XX : push imm32"},
    { 5, {0xb8, 0x71, 0x72, 0x73, 0x74}, 0, "b8 XX XX XX XX : mov eax, XX XX XX XX"},
    { 5, {0xB9, 0x71, 0x72, 0x73, 0x74}, 0, "b9 XX XX XX XX : mov ecx, XX XX XX XX"},
#if SANITIZER_WINDOWS_x64
    // sorted list
    { 2, {0x40, 0x50}, 0, "40 50 : push rax"},
    { 2, {0x40, 0x51}, 0, "40 51 : push rcx"},
    { 2, {0x40, 0x52}, 0, "40 52 : push rdx"},
    { 2, {0x40, 0x53}, 0, "40 53 : push rbx"},
    { 2, {0x40, 0x54}, 0, "40 54 : push rsp"},
    { 2, {0x40, 0x55}, 0, "40 55 : push rbp"},
    { 2, {0x40, 0x56}, 0, "40 56 : push rsi"},
    { 2, {0x40, 0x57}, 0, "40 57 : push rdi"},
    { 2, {0x41, 0x54}, 0, "41 54 : push r12"},
    { 2, {0x41, 0x55}, 0, "41 55 : push r13"},
    { 2, {0x41, 0x56}, 0, "41 56 : push r14"},
    { 2, {0x41, 0x57}, 0, "41 57 : push r15"},
    { 2, {0x66, 0x90}, 0, "66 90 : Two-byte NOP"},
    { 2, {0x84, 0xc0}, 0, "84 c0 : test al, al"},
    { 2, {0x8a, 0x01}, 0, "8a 01 : mov al, byte ptr [rcx]"},
    { 3, {0x0f, 0xb6, 0xc2}, 0, "0f b6 c2 : movzx eax, dl"},
    { 3, {0x0f, 0xb6, 0xd2}, 0, "0f b6 d2 : movzx edx, dl"},
    { 3, {0x0f, 0xb7, 0x10}, 0, "0f b7 10 : movzx edx, WORD PTR [rax]"},
    { 3, {0x41, 0x8b, 0xc0}, 0, "41 8b c0 : mov eax, r8d"},
    { 3, {0x41, 0x8b, 0xc1}, 0, "41 8b c1 : mov eax, r9d"},
    { 3, {0x41, 0x8b, 0xc2}, 0, "41 8b c2 : mov eax, r10d"},
    { 3, {0x41, 0x8b, 0xc3}, 0, "41 8b c3 : mov eax, r11d"},
    { 3, {0x41, 0x8b, 0xc4}, 0, "41 8b c4 : mov eax, r12d"},
    { 3, {0x45, 0x33, 0xc0}, 0, "45 33 c0 : xor r8d, r8d"},
    { 3, {0x45, 0x33, 0xc9}, 0, "45 33 c9 : xor r9d, r9d"},
    { 3, {0x45, 0x33, 0xdb}, 0, "45 33 db : xor r11d, r11d"},
    { 3, {0x48, 0x2b, 0xca}, 0, "48 2b ca : sub rcx, rdx"},
    { 3, {0x48, 0x2b, 0xd1}, 0, "48 2b d1 : sub rdx, rcx"},
    { 3, {0x48, 0x3b, 0xca}, 0, "48 3b ca : cmp rcx, rdx"},
    { 3, {0x48, 0x85, 0xc0}, 0, "48 85 c0 : test rax, rax"},
    { 3, {0x48, 0x85, 0xc9}, 0, "48 85 c9 : test rcx, rcx"},
    { 3, {0x48, 0x85, 0xd2}, 0, "48 85 d2 : test rdx, rdx"},
    { 3, {0x48, 0x85, 0xdb}, 0, "48 85 db : test rbx, rbx"},
    { 3, {0x48, 0x85, 0xe4}, 0, "48 85 e4 : test rsp, rsp"},
    { 3, {0x48, 0x85, 0xed}, 0, "48 85 ed : test rbp, rbp"},
    { 3, {0x48, 0x89, 0xe5}, 0, "48 89 e5 : mov rbp, rsp"},
    { 3, {0x48, 0x8b, 0xc1}, 0, "48 8b c1 : mov rax, rcx"},
    { 3, {0x48, 0x8b, 0xc4}, 0, "48 8b c4 : mov rax, rsp"},
    { 3, {0x48, 0x8b, 0xd1}, 0, "48 8b d1 : mov rdx, rcx"},
    { 3, {0x48, 0xf7, 0xd9}, 0, "48 f7 d9 : neg rcx"},
    { 3, {0x48, 0xff, 0xc0}, 0, "48 ff c0 : inc rax"},
    { 3, {0x48, 0xff, 0xc1}, 0, "48 ff c1 : inc rcx"},
    { 3, {0x48, 0xff, 0xc2}, 0, "48 ff c2 : inc rdx"},
    { 3, {0x48, 0xff, 0xc3}, 0, "48 ff c3 : inc rbx"},
    { 3, {0x48, 0xff, 0xc6}, 0, "48 ff c6 : inc rsi"},
    { 3, {0x48, 0xff, 0xc7}, 0, "48 ff c7 : inc rdi"},
    { 3, {0x49, 0xff, 0xc0}, 0, "49 ff c0 : inc r8"},
    { 3, {0x49, 0xff, 0xc1}, 0, "49 ff c1 : inc r9"},
    { 3, {0x49, 0xff, 0xc2}, 0, "49 ff c2 : inc r10"},
    { 3, {0x49, 0xff, 0xc3}, 0, "49 ff c3 : inc r11"},
    { 3, {0x49, 0xff, 0xc4}, 0, "49 ff c4 : inc r12"},
    { 3, {0x49, 0xff, 0xc5}, 0, "49 ff c5 : inc r13"},
    { 3, {0x49, 0xff, 0xc6}, 0, "49 ff c6 : inc r14"},
    { 3, {0x49, 0xff, 0xc7}, 0, "49 ff c7 : inc r15"},
    { 3, {0x4c, 0x8b, 0xc1}, 0, "4c 8b c1 : mov r8, rcx"},
    { 3, {0x4c, 0x8b, 0xc9}, 0, "4c 8b c9 : mov r9, rcx"},
    { 3, {0x4c, 0x8b, 0xd1}, 0, "4c 8b d1 : mov r10, rcx"},
    { 3, {0x4c, 0x8b, 0xd2}, 0, "4c 8b d2 : mov r10, rdx"},
    { 3, {0x4c, 0x8b, 0xd9}, 0, "4c 8b d9 : mov r11, rcx"},
    { 3, {0x4c, 0x8b, 0xdc}, 0, "4c 8b dc : mov r11, rsp"},
    { 3, {0x4d, 0x0b, 0xc0}, 0, "4d 0b c0 : or r8, r8"},
    { 3, {0x4d, 0x85, 0xc0}, 0, "4d 85 c0 : test r8, r8"},
    { 3, {0x4d, 0x85, 0xc9}, 0, "4d 85 c9 : test r9, r9"},
    { 3, {0x4d, 0x85, 0xd2}, 0, "4d 85 d2 : test r10, r10"},
    { 3, {0x4d, 0x85, 0xdb}, 0, "4d 85 db : test r11, r11"},
    { 3, {0x4d, 0x85, 0xe4}, 0, "4d 85 e4 : test r12, r12"},
    { 3, {0x4d, 0x85, 0xed}, 0, "4d 85 ed : test r13, r13"},
    { 3, {0x4d, 0x85, 0xf6}, 0, "4d 85 f6 : test r14, r14"},
    { 3, {0x4d, 0x85, 0xff}, 0, "4d 85 ff : test r15, r15"},
    { 3, {0xf6, 0xc1, 0x72}, 0, "f6 c1 XX : test cl, XX"},
    { 4, {0x44, 0x0f, 0xb6, 0x1a}, 0, "44 0f b6 1a : movzx r11d, BYTE PTR [rdx]"},
    { 4, {0x44, 0x8d, 0x42, 0x73}, 0, "44 8d 42 XX : lea r8d , [rdx + XX]"},
    { 4, {0x48, 0x83, 0xec, 0x73}, 0, "48 83 ec XX : sub rsp, XX"},
    { 4, {0x48, 0x89, 0x58, 0x73}, 0, "48 89 58 XX : mov QWORD PTR[rax + XX], rbx"},
    { 4, {0x49, 0x83, 0xf8, 0x73}, 0, "49 83 f8 XX : cmp r8, XX"},
    { 4, {0x80, 0x78, 0x72, 0x73}, 0, "80 78 YY XX : cmp BYTE PTR [rax+YY], XX"},
    { 4, {0x80, 0x79, 0x72, 0x73}, 0, "80 79 YY XX : cmp BYTE ptr [rcx+YY], XX"},
    { 4, {0x80, 0x7A, 0x72, 0x73}, 0, "80 7A YY XX : cmp BYTE PTR [rdx+YY], XX"},
    { 4, {0x80, 0x7B, 0x72, 0x73}, 0, "80 7B YY XX : cmp BYTE PTR [rbx+YY], XX"},
    { 4, {0x80, 0x7D, 0x72, 0x73}, 0, "80 7D YY XX : cmp BYTE PTR [rbp+YY], XX"},
    { 4, {0x80, 0x7E, 0x72, 0x73}, 0, "80 7E YY XX : cmp BYTE PTR [rsi+YY], XX"},
    { 4, {0x89, 0x54, 0x24, 0x73}, 0, "89 54 24 XX : mov DWORD PTR[rsp + XX], edx"},
    { 5, {0x44, 0x89, 0x44, 0x24, 0x74}, 0, "44 89 44 24 XX : mov DWORD PTR [rsp + XX], r8d"},
    { 5, {0x44, 0x89, 0x4c, 0x24, 0x74}, 0, "44 89 4c 24 XX : mov DWORD PTR [rsp + XX], r9d"},
    { 5, {0x48, 0x89, 0x4C, 0x24, 0x74}, 0, "48 89 4C 24 XX : mov QWORD PTR [rsp + XX], rcx"},
    { 5, {0x48, 0x89, 0x54, 0x24, 0x74}, 0, "48 89 54 24 XX : mov QWORD PTR [rsp + XX], rdx"},
    { 5, {0x48, 0x89, 0x5c, 0x24, 0x74}, 0, "48 89 5c 24 XX : mov QWORD PTR [rsp + XX], rbx"},
    { 5, {0x48, 0x89, 0x6c, 0x24, 0x74}, 0, "48 89 6C 24 XX : mov QWORD ptr [rsp + XX], rbp"},
    { 5, {0x48, 0x89, 0x74, 0x24, 0x74}, 0, "48 89 74 24 XX : mov QWORD PTR [rsp + XX], rsi"},
    { 5, {0x48, 0x89, 0x7c, 0x24, 0x74}, 0, "48 89 7c 24 XX : mov QWORD PTR [rsp + XX], rdi"},
    { 5, {0x48, 0x8b, 0x44, 0x24, 0x74}, 0, "48 8b 44 24 XX : mov rax, QWORD ptr [rsp + XX]"},
    { 5, {0x48, 0x8d, 0x6c, 0x24, 0x74}, 0, "48 8d 6c 24 XX : lea rbp, [rsp + XX]"},
    { 5, {0x4c, 0x89, 0x44, 0x24, 0x74}, 0, "4c 89 44 24 XX : mov QWORD PTR [rsp + XX], r8"},
    { 5, {0x4c, 0x89, 0x4c, 0x24, 0x74}, 0, "4c 89 4c 24 XX : mov QWORD PTR [rsp + XX], r9"},
    { 5, {0x83, 0x44, 0x72, 0x73, 0x74}, 0, "83 44 72 XX YY : add DWORD PTR [rdx+rsi*2+XX],YY"},
    { 5, {0x83, 0x64, 0x24, 0x73, 0x74}, 0, "83 64 24 XX YY : and DWORD PTR [rsp+XX], YY"},
    { 6, {0x48, 0x83, 0x64, 0x24, 0x74, 0x75}, 0, "48 83 64 24 XX YY : and QWORD PTR [rsp + XX], YY"},
    { 6, {0x66, 0x81, 0x78, 0x73, 0x74, 0x75}, 0, "66 81 78 XX YY YY : cmp WORD PTR [rax+XX], YY YY"},
    { 6, {0x66, 0x81, 0x79, 0x73, 0x74, 0x75}, 0, "66 81 79 XX YY YY : cmp WORD PTR [rcx+XX], YY YY"},
    { 6, {0x66, 0x81, 0x7a, 0x73, 0x74, 0x75}, 0, "66 81 7a XX YY YY : cmp WORD PTR [rdx+XX], YY YY"},
    { 6, {0x66, 0x81, 0x7b, 0x73, 0x74, 0x75}, 0, "66 81 7b XX YY YY : cmp WORD PTR [rbx+XX], YY YY"},
    { 6, {0x66, 0x81, 0x7e, 0x73, 0x74, 0x75}, 0, "66 81 7e XX YY YY : cmp WORD PTR [rsi+XX], YY YY"},
    { 6, {0x66, 0x81, 0x7f, 0x73, 0x74, 0x75}, 0, "66 81 7f XX YY YY : cmp WORD PTR [rdi+XX], YY YY"},
    { 6, {0x8A, 0x05, 0x72, 0x73, 0x74, 0x75}, 2, "8A 05 XX XX XX XX : mov al, byte ptr [XX XX XX XX]"},
    { 6, {0x8B, 0x05, 0x72, 0x73, 0x74, 0x75}, 2, "8B 05 XX XX XX XX : mov eax, dword ptr [XX XX XX XX]"},
    { 6, {0xF2, 0x0f, 0x11, 0x44, 0x24, 0x75}, 0, "f2 0f 11 44 24 XX : movsd QWORD PTR [rsp + XX], xmm0"},
    { 6, {0xF2, 0x0f, 0x11, 0x4c, 0x24, 0x75}, 0, "f2 0f 11 4c 24 XX : movsd QWORD PTR [rsp + XX], xmm1"},
    { 6, {0xF2, 0x0f, 0x11, 0x54, 0x24, 0x75}, 0, "f2 0f 11 54 24 XX : movsd QWORD PTR [rsp + XX], xmm2"},
    { 6, {0xF2, 0x0f, 0x11, 0x5c, 0x24, 0x75}, 0, "f2 0f 11 5c 24 XX : movsd QWORD PTR [rsp + XX], xmm3"},
    { 6, {0xF2, 0x0f, 0x11, 0x64, 0x24, 0x75}, 0, "f2 0f 11 64 24 XX : movsd QWORD PTR [rsp + XX], xmm4"},
    { 7, {0x48, 0x81, 0xec, 0x73, 0x74, 0x75, 0x76}, 0, "48 81 EC XX XX XX XX : sub rsp, XXXXXXXX"},
    { 7, {0x48, 0x89, 0x0d, 0x73, 0x74, 0x75, 0x76}, 3, "48 89 0d XX XX XX XX : mov QWORD PTR [rip + XXXXXXXX], rcx"},
    { 7, {0x48, 0x89, 0x15, 0x73, 0x74, 0x75, 0x76}, 3, "48 89 15 XX XX XX XX : mov QWORD PTR [rip + XXXXXXXX], rdx"},
    { 7, {0x48, 0x8b, 0x05, 0x73, 0x74, 0x75, 0x76}, 3, "48 8b 05 XX XX XX XX : mov rax, QWORD PTR [rip + XXXXXXXX]"},
    { 7, {0x48, 0x8d, 0x05, 0x73, 0x74, 0x75, 0x76}, 3, "48 8d 05 XX XX XX XX : lea rax, QWORD PTR [rip + XXXXXXXX]"},
    { 7, {0x48, 0xff, 0x25, 0x73, 0x74, 0x75, 0x76}, 3, "48 ff 25 XX XX XX XX : rex.W jmp QWORD PTR [rip + XXXXXXXX]"},
    { 7, {0x4C, 0x8D, 0x15, 0x73, 0x74, 0x75, 0x76}, 3, "4c 8d 15 XX XX XX XX : lea r10, [rip + XX]"},
    { 7, {0x81, 0x78, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "81 78 YY XX XX XX XX : cmp DWORD PTR [rax+YY], XX XX XX XX"},
    { 7, {0x81, 0x79, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "81 79 YY XX XX XX XX : cmp dword ptr [rcx+YY], XX XX XX XX"},
    { 7, {0x81, 0x7A, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "81 7A YY XX XX XX XX : cmp DWORD PTR [rdx+YY], XX XX XX XX"},
    { 7, {0x81, 0x7B, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "81 7B YY XX XX XX XX : cmp DWORD PTR [rbx+YY], XX XX XX XX"},
    { 7, {0x81, 0x7D, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "81 7D YY XX XX XX XX : cmp DWORD PTR [rbp+YY], XX XX XX XX"},
    { 7, {0x81, 0x7E, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "81 7E YY XX XX XX XX : cmp DWORD PTR [rsi+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x78, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 78 XX YY YY YY YY : cmp DWORD PTR [r8+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x79, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 79 XX YY YY YY YY : cmp DWORD PTR [r9+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x7a, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 7a XX YY YY YY YY : cmp DWORD PTR [r10+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x7b, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 7b XX YY YY YY YY : cmp DWORD PTR [r11+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x7d, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 7d XX YY YY YY YY : cmp DWORD PTR [r13+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x7e, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 7e XX YY YY YY YY : cmp DWORD PTR [r14+YY], XX XX XX XX"},
    { 8, {0x41, 0x81, 0x7f, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "41 81 7f YY XX XX XX XX : cmp DWORD PTR [r15+YY], XX XX XX XX"},
    { 8, {0x81, 0x7c, 0x24, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "81 7c 24 YY XX XX XX XX : cmp DWORD PTR [rsp+YY], XX XX XX XX"},
    { 8, {0xc7, 0x44, 0x24, 0x73, 0x74, 0x75, 0x76, 0x77}, 0, "C7 44 24 XX YY YY YY YY : mov dword ptr [rsp + XX], YYYYYYYY"},
    { 9, {0x41, 0x81, 0x7c, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78}, 0, "41 81 7c ZZ YY XX XX XX XX : cmp DWORD PTR [reg+reg*n+YY], XX XX XX XX"},
    { 9, {0xA1, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78}, 0, "A1 XX XX XX XX XX XX XX XX : movabs eax, dword ptr ds:[XXXXXXXX]"},
#else
    // sorted list
    { 3, {0x8B, 0x45, 0x72}, 0, "8B 45 XX : mov eax, dword ptr [ebp + XX]"},
    { 3, {0x8B, 0x5D, 0x72}, 0, "8B 5D XX : mov ebx, dword ptr [ebp + XX]"},
    { 3, {0x8B, 0x75, 0x72}, 0, "8B 75 XX : mov esi, dword ptr [ebp + XX]"},
    { 3, {0x8B, 0x7D, 0x72}, 0, "8B 7D XX : mov edi, dword ptr [ebp + XX]"},
    { 3, {0xFF, 0x75, 0x72}, 0, "FF 75 XX : push dword ptr [ebp + XX]"},
    { 4, {0x83, 0x7D, 0x72, 0x73}, 0, "83 7D XX YY : cmp dword ptr [ebp + XX], YY"},
    { 4, {0x8A, 0x44, 0x24, 0x73}, 0, "8A 44 24 XX : mov eal, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x44, 0x24, 0x73}, 0, "8B 44 24 XX : mov eax, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x4C, 0x24, 0x73}, 0, "8B 4C 24 XX : mov ecx, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x54, 0x24, 0x73}, 0, "8B 54 24 XX : mov edx, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x5C, 0x24, 0x73}, 0, "8B 5C 24 XX : mov ebx, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x6C, 0x24, 0x73}, 0, "8B 6C 24 XX : mov ebp, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x74, 0x24, 0x73}, 0, "8B 74 24 XX : mov esi, dword ptr [esp + XX]"},
    { 4, {0x8B, 0x7C, 0x24, 0x73}, 0, "8B 7C 24 XX : mov edi, dword ptr [esp + XX]"},
    { 5, {0x0F, 0xB6, 0x44, 0x24, 0x74}, 0, "0F B6 44 24 XX : movzx eax, byte ptr [esp + XX]"},
    { 5, {0xA1, 0x71, 0x72, 0x73, 0x74}, 0, "A1 XX XX XX XX : mov eax, dword ptr ds:[XXXXXXXX]"},
    { 6, {0xF7, 0xC1, 0x72, 0x73, 0x74, 0x75}, 0, "F7 C1 XX YY ZZ WW : test ecx, WWZZYYXX"},
    { 7, {0x83, 0x3D, 0x72, 0x73, 0x74, 0x75, 0x76}, 0, "83 3D XX YY ZZ WW TT : cmp TT, WWZZYYXX"},
#endif
    // clang-format on
};

std::string dumpInstruction(unsigned arrayIndex,
                            const InstructionSizeData &data) {
  std::stringstream ret;
  ret << "  with arrayIndex=" << arrayIndex << " {";
  for (size_t i = 0; i < data.size; i++) {
    if (i > 0)
      ret << ", ";
    ret << "0x" << std::setfill('0') << std::setw(2) << std::right << std::hex
        << (int)data.instr[i];
  }
  ret << "} " << data.comment;
  return ret.str();
}

TEST(Interception, GetInstructionSize) {
  for (unsigned i = 0; i < sizeof(data) / sizeof(*data); i++) {
    size_t rel_offset = ~0L;
    size_t size = __interception::TestOnlyGetInstructionSize(
        (uptr)data[i].instr, &rel_offset);
    EXPECT_EQ(data[i].size, size) << dumpInstruction(i, data[i]);
    EXPECT_EQ(data[i].rel_offset, rel_offset) << dumpInstruction(i, data[i]);
  }
}

}  // namespace __interception

#    endif  // !SANITIZER_WINDOWS_ARM64
#endif  // SANITIZER_WINDOWS
#endif  // #if !SANITIZER_DEBUG
