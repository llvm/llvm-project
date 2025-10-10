//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _Unwind_Backtrace() walks up from a signal handler and produces
// a correct traceback when the function raising the signal does not save
// the link register or does not store the stack back chain.

// REQUIRES: target={{.+}}-aix{{.*}}

// Test when the function raising the signal does not save the link register
// RUN: %{cxx} -x c++ %s -o %t.exe -DCXX_CODE %{flags} %{compile_flags}
// RUN: %{exec} %t.exe

// Test when the function raising the signal does not store stack back chain.
// RUN: %{cxx} -x c++ -c %s -o %t1.o -DCXX_CODE -DNOBACKCHAIN %{flags} \
// RUN:   %{compile_flags}
// RUN: %{cxx} -c %s -o %t2.o %{flags} %{compile_flags}
// RUN: %{cxx} -o %t1.exe %t1.o %t2.o %{flags} %{link_flags}
// RUN: %{exec} %t1.exe

#ifdef CXX_CODE

#undef NDEBUG
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/debug.h>
#include <unwind.h>

#define NAME_ARRAY_SIZE 10
#define NAMES_EXPECTED   6

const char* namesExpected[] = {"handler", "abc", "bar", "foo", "main",
                               "__start"};
char *namesObtained[NAME_ARRAY_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

int funcIndex = 0;

// Get the function name from traceback table.
char *getFuncName(uintptr_t pc, uint16_t *nameLen) {
  uint32_t *p = reinterpret_cast<uint32_t *>(pc);

  // Keep looking forward until a word of 0 is found. The traceback
  // table starts at the following word.
  while (*p)
    ++p;
  tbtable *TBTable = reinterpret_cast<tbtable *>(p + 1);

  if (!TBTable->tb.name_present)
    return NULL;

  // Get to the optional portion of the traceback table.
  p = reinterpret_cast<uint32_t *>(&TBTable->tb_ext);

  // Skip field parminfo if it exists.
  if (TBTable->tb.fixedparms || TBTable->tb.floatparms)
    ++p;

  // Skip field tb_offset if it exists.
  if (TBTable->tb.has_tboff)
    ++p;

  // Skip field hand_mask if it exists.
  if (TBTable->tb.int_hndl)
    ++p;

  // Skip fields ctl_info and ctl_info_disp if they exist.
  if (TBTable->tb.has_ctl)
    p += 1 + *p;

  *nameLen = *reinterpret_cast<uint16_t *>(p);
  return reinterpret_cast<char *>(p) + sizeof(uint16_t);
}

_Unwind_Reason_Code callBack(struct _Unwind_Context *uc, void *arg) {
  (void)arg;
  uint16_t nameLen;
  uintptr_t ip = _Unwind_GetIP(uc);
  if (funcIndex < NAME_ARRAY_SIZE)
    namesObtained[funcIndex++] = strndup(getFuncName(ip, &nameLen), nameLen);
  return _URC_NO_REASON;
}

extern "C" void handler(int signum) {
  (void)signum;
  // Walk stack frames for traceback.
  _Unwind_Backtrace(callBack, NULL);

  // Verify the traceback.
  assert(funcIndex <= NAMES_EXPECTED && "Obtained names more than expected");
  for (int i = 0; i < funcIndex; ++i) {
    assert(!strcmp(namesExpected[i], namesObtained[i]) &&
           "Function names do not match");
    free(namesObtained[i]);
  }
  exit(0);
}

#ifdef NOBACKCHAIN
// abc() is in assembly. It raises signal SIGSEGV and does not store
// the stack back chain.
extern "C" void abc();

#else
volatile int *null = 0;

// abc() raises signal SIGSEGV and does not save the link register.
extern "C" __attribute__((noinline)) void abc() {
  // Produce a SIGSEGV.
  *null = 0;
}
#endif

extern "C" __attribute__((noinline)) void bar() {
  abc();
}

extern "C" __attribute__((noinline)) void foo() {
  bar();
}

int main(int, char**) {
  // Set signal handler for SIGSEGV.
  signal(SIGSEGV, handler);
  foo();
  return 0;
}

#else // Assembly code for abc().
// This assembly code is similar to the following C code but it saves the
// link register.
//
// int *badp = 0;
// void abc() {
//   *badp = 0;
// }

#ifdef __64BIT__
        .csect [PR],5
        .file   "abc.c"
        .globl  abc[DS]                 # -- Begin function abc
        .globl  .abc
        .align  4
        .csect abc[DS],3
        .vbyte  8, .abc                 # @abc
        .vbyte  8, TOC[TC0]
        .vbyte  8, 0
        .csect [PR],5
.abc:
# %bb.0:                                # %entry
        mflr 0
        std 0, 16(1)
        ld 3, L..C0(2)                  # @badp
        bl $+4
        ld 4, 0(3)
        li 3, 0
        stw 3, 0(4)
        ld 0, 16(1)
        mtlr 0
        blr
L..abc0:
        .vbyte  4, 0x00000000           # Traceback table begin
        .byte   0x00                    # Version = 0
        .byte   0x09                    # Language = CPlusPlus
        .byte   0x20                    # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
                                        # +HasTraceBackTableOffset, -IsInternalProcedure
                                        # -HasControlledStorage, -IsTOCless
                                        # -IsFloatingPointPresent
                                        # -IsFloatingPointOperationLogOrAbortEnabled
        .byte   0x61                    # -IsInterruptHandler, +IsFunctionNamePresent, +IsAllocaUsed
                                        # OnConditionDirective = 0, -IsCRSaved, +IsLRSaved
        .byte   0x00                    # -IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
        .byte   0x01                    # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
        .byte   0x00                    # NumberOfFixedParms = 0
        .byte   0x01                    # NumberOfFPParms = 0, +HasParmsOnStack
        .vbyte  4, L..abc0-.abc         # Function size
        .vbyte  2, 0x0003               # Function name len = 3
        .byte   "abc"                   # Function Name
        .byte   0x1f                    # AllocaUsed
                                        # -- End function
        .csect badp[RW],3
        .globl  badp[RW]                # @badp
        .align  3
        .vbyte  8, 0
        .toc
L..C0:
        .tc badp[TC],badp[RW]
#else
        .csect [PR],5
        .file   "abc.c"
        .globl  abc[DS]                 # -- Begin function abc
        .globl  .abc
        .align  4
        .csect abc[DS],2
        .vbyte  4, .abc                 # @abc
        .vbyte  4, TOC[TC0]
        .vbyte  4, 0
        .csect [PR],5
.abc:
# %bb.0:                                # %entry
        mflr 0
        stw 0, 8(1)
        lwz 3, L..C0(2)                 # @badp
        bl $+4
        lwz 4, 0(3)
        li 3, 0
        stw 3, 0(4)
        lwz 0, 8(1)
        mtlr 0
        blr
L..abc0:
        .vbyte  4, 0x00000000           # Traceback table begin
        .byte   0x00                    # Version = 0
        .byte   0x09                    # Language = CPlusPlus
        .byte   0x20                    # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
                                        # +HasTraceBackTableOffset, -IsInternalProcedure
                                        # -HasControlledStorage, -IsTOCless
                                        # -IsFloatingPointPresent
                                        # -IsFloatingPointOperationLogOrAbortEnabled
        .byte   0x61                    # -IsInterruptHandler, +IsFunctionNamePresent, +IsAllocaUsed
                                        # OnConditionDirective = 0, -IsCRSaved, +IsLRSaved
        .byte   0x00                    # -IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
        .byte   0x01                    # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
        .byte   0x00                    # NumberOfFixedParms = 0
        .byte   0x01                    # NumberOfFPParms = 0, +HasParmsOnStack
        .vbyte  4, L..abc0-.abc         # Function size
        .vbyte  2, 0x0003               # Function name len = 3
        .byte   "abc"                   # Function Name
        .byte   0x1f                    # AllocaUsed
                                        # -- End function
        .csect badp[RW],2
        .globl  badp[RW]                # @badp
        .align  2
        .vbyte  4, 0
        .toc
L..C0:
        .tc badp[TC],badp[RW]
#endif // __64BIT__
#endif // CXX_CODE
