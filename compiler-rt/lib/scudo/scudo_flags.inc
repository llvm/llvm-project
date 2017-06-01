//===-- scudo_flags.inc -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Hardened Allocator runtime flags.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_FLAG
# error "Define SCUDO_FLAG prior to including this file!"
#endif

SCUDO_FLAG(int, QuarantineSizeMb, 64,
           "Size (in Mb) of quarantine used to delay the actual deallocation "
           "of chunks. Lower value may reduce memory usage but decrease the "
           "effectiveness of the mitigation.")

SCUDO_FLAG(int, ThreadLocalQuarantineSizeKb, 1024,
          "Size (in Kb) of per-thread cache used to offload the global "
          "quarantine. Lower value may reduce memory usage but might increase "
          "the contention on the global quarantine.")

SCUDO_FLAG(bool, DeallocationTypeMismatch, true,
          "Report errors on malloc/delete, new/free, new/delete[], etc.")

SCUDO_FLAG(bool, DeleteSizeMismatch, true,
           "Report errors on mismatch between size of new and delete.")

SCUDO_FLAG(bool, ZeroContents, false,
          "Zero chunk contents on allocation and deallocation.")
