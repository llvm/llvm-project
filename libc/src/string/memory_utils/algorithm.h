//===-- Algorithms to compose sized memory operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Higher order primitives that build upon the SizedOpT facility.
// They constitute the basic blocks for composing memory functions.
// This file defines the following operations:
// - Skip
// - Tail
// - HeadTail
// - Loop
// - Align
//
// See each class for documentation.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ALGORITHM_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ALGORITHM_H

#include "src/string/memory_utils/address.h" // Address
#include "src/string/memory_utils/utils.h"   // offset_to_next_aligned

#include <stddef.h> // ptrdiff_t

namespace __llvm_libc {

// We are not yet allowed to use asserts in low level memory operations as
// assert itself could depend on them.
// We define this empty macro so we can enable them as soon as possible and keep
// track of invariants.
#define LIBC_ASSERT(COND)

// An operation that allows to skip the specified amount of bytes.
template <ptrdiff_t Bytes> struct Skip {
  template <typename NextT> struct Then {
    template <typename DstAddrT>
    static inline void set(DstAddrT dst, ubyte value) {
      static_assert(NextT::IS_FIXED_SIZE);
      NextT::set(offsetAddr<Bytes>(dst), value);
    }

    template <typename SrcAddrT1, typename SrcAddrT2>
    static inline uint64_t isDifferent(SrcAddrT1 src1, SrcAddrT2 src2) {
      static_assert(NextT::IS_FIXED_SIZE);
      return NextT::isDifferent(offsetAddr<Bytes>(src1),
                                offsetAddr<Bytes>(src2));
    }

    template <typename SrcAddrT1, typename SrcAddrT2>
    static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2) {
      static_assert(NextT::IS_FIXED_SIZE);
      return NextT::threeWayCmp(offsetAddr<Bytes>(src1),
                                offsetAddr<Bytes>(src2));
    }

    template <typename SrcAddrT1, typename SrcAddrT2>
    static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2,
                                      size_t runtime_size) {
      static_assert(NextT::IS_RUNTIME_SIZE);
      return NextT::threeWayCmp(offsetAddr<Bytes>(src1),
                                offsetAddr<Bytes>(src2), runtime_size - Bytes);
    }
  };
};

// Compute the address of a tail operation.
// Because of the runtime size, we loose the alignment information.
template <size_t Size, typename AddrT>
static auto tailAddr(AddrT addr, size_t runtime_size) {
  static_assert(IsAddressType<AddrT>::Value);
  return offsetAddrAssumeAligned<1>(addr, runtime_size - Size);
}

// Perform the operation on the last 'Size' bytes of the buffer.
//
// e.g. with
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [________XXXXXXXX___]
//
// Precondition: `runtime_size >= Size`.
template <typename SizedOpT> struct Tail {
  static_assert(SizedOpT::IS_FIXED_SIZE);
  static constexpr bool IS_RUNTIME_SIZE = true;
  static constexpr size_t SIZE = SizedOpT::SIZE;

  template <typename DstAddrT, typename SrcAddrT>
  static inline void copy(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
    SizedOpT::copy(tailAddr<SIZE>(dst, runtime_size),
                   tailAddr<SIZE>(src, runtime_size));
  }

  template <typename DstAddrT, typename SrcAddrT>
  static inline void move(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
    SizedOpT::move(tailAddr<SIZE>(dst, runtime_size),
                   tailAddr<SIZE>(src, runtime_size));
  }

  template <typename DstAddrT>
  static inline void set(DstAddrT dst, ubyte value, size_t runtime_size) {
    SizedOpT::set(tailAddr<SIZE>(dst, runtime_size), value);
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline uint64_t isDifferent(SrcAddrT1 src1, SrcAddrT2 src2,
                                     size_t runtime_size) {
    return SizedOpT::isDifferent(tailAddr<SIZE>(src1, runtime_size),
                                 tailAddr<SIZE>(src2, runtime_size));
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2,
                                    size_t runtime_size) {
    return SizedOpT::threeWayCmp(tailAddr<SIZE>(src1, runtime_size),
                                 tailAddr<SIZE>(src2, runtime_size));
  }
};

// Perform the operation on the first and the last `SizedOpT::Size` bytes of the
// buffer. This is useful for overlapping operations.
//
// e.g. with
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [__XXXXXXXX_________]
// [________XXXXXXXX___]
//
// Precondition: `runtime_size >= Size && runtime_size <= 2 x Size`.
template <typename SizedOpT> struct HeadTail {
  static_assert(SizedOpT::IS_FIXED_SIZE);
  static constexpr bool IS_RUNTIME_SIZE = true;

  template <typename DstAddrT, typename SrcAddrT>
  static inline void copy(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
    LIBC_ASSERT(runtime_size >= SizedOpT::SIZE);
    SizedOpT::copy(dst, src);
    Tail<SizedOpT>::copy(dst, src, runtime_size);
  }

  template <typename DstAddrT, typename SrcAddrT>
  static inline void move(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
    LIBC_ASSERT(runtime_size >= SizedOpT::SIZE);
    static constexpr size_t BLOCK_SIZE = SizedOpT::SIZE;
    // The load and store operations can be performed in any order as long as
    // they are not interleaved. More investigations are needed to determine the
    // best order.
    auto head = SizedOpT::load(src);
    auto tail = SizedOpT::load(tailAddr<BLOCK_SIZE>(src, runtime_size));
    SizedOpT::store(tailAddr<BLOCK_SIZE>(dst, runtime_size), tail);
    SizedOpT::store(dst, head);
  }

  template <typename DstAddrT>
  static inline void set(DstAddrT dst, ubyte value, size_t runtime_size) {
    LIBC_ASSERT(runtime_size >= SizedOpT::SIZE);
    SizedOpT::set(dst, value);
    Tail<SizedOpT>::set(dst, value, runtime_size);
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline uint64_t isDifferent(SrcAddrT1 src1, SrcAddrT2 src2,
                                     size_t runtime_size) {
    LIBC_ASSERT(runtime_size >= SizedOpT::SIZE);
    // Two strategies can be applied here:
    // 1. Compute head and tail and compose them with a bitwise or operation.
    // 2. Stop early if head is different.
    // We chose the later because HeadTail operations are typically performed
    // with sizes ranging from 4 to 256 bytes. The cost of the loads is then
    // significantly larger than the cost of the branch.
    if (const uint64_t res = SizedOpT::isDifferent(src1, src2))
      return res;
    return Tail<SizedOpT>::isDifferent(src1, src2, runtime_size);
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2,
                                    size_t runtime_size) {
    LIBC_ASSERT(runtime_size >= SizedOpT::SIZE &&
                runtime_size <= 2 * SizedOpT::SIZE);
    if (const int32_t res = SizedOpT::threeWayCmp(src1, src2))
      return res;
    return Tail<SizedOpT>::threeWayCmp(src1, src2, runtime_size);
  }
};

// Simple loop ending with a Tail operation.
//
// e.g. with
// [12345678123456781234567812345678]
// [__XXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
// [__XXXXXXXX_______________________]
// [__________XXXXXXXX_______________]
// [__________________XXXXXXXX_______]
// [______________________XXXXXXXX___]
//
// Precondition:
// - runtime_size >= Size
template <typename SizedOpT> struct Loop {
  static_assert(SizedOpT::IS_FIXED_SIZE);
  static constexpr bool IS_RUNTIME_SIZE = true;
  static constexpr size_t BLOCK_SIZE = SizedOpT::SIZE;

  template <typename DstAddrT, typename SrcAddrT>
  static inline void copy(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
    size_t offset = 0;
    do {
      SizedOpT::copy(offsetAddrMultiplesOf<BLOCK_SIZE>(dst, offset),
                     offsetAddrMultiplesOf<BLOCK_SIZE>(src, offset));
      offset += BLOCK_SIZE;
    } while (offset < runtime_size - BLOCK_SIZE);
    Tail<SizedOpT>::copy(dst, src, runtime_size);
  }

  // Move forward suitable when dst < src. We load the tail bytes before
  // handling the loop.
  //
  // e.g. Moving two bytes
  // [   |       |       |       |       |]
  // [___XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
  // [_________________________LLLLLLLL___]
  // [___LLLLLLLL_________________________]
  // [_SSSSSSSS___________________________]
  // [___________LLLLLLLL_________________]
  // [_________SSSSSSSS___________________]
  // [___________________LLLLLLLL_________]
  // [_________________SSSSSSSS___________]
  // [_______________________SSSSSSSS_____]
  template <typename DstAddrT, typename SrcAddrT>
  static inline void move(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
    const auto tail_value =
        SizedOpT::load(tailAddr<BLOCK_SIZE>(src, runtime_size));
    size_t offset = 0;
    do {
      SizedOpT::move(offsetAddrMultiplesOf<BLOCK_SIZE>(dst, offset),
                     offsetAddrMultiplesOf<BLOCK_SIZE>(src, offset));
      offset += BLOCK_SIZE;
    } while (offset < runtime_size - BLOCK_SIZE);
    SizedOpT::store(tailAddr<BLOCK_SIZE>(dst, runtime_size), tail_value);
  }

  // Move backward suitable when dst > src. We load the head bytes before
  // handling the loop.
  //
  // e.g. Moving two bytes
  // [   |       |       |       |       |]
  // [___XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
  // [___LLLLLLLL_________________________]
  // [_________________________LLLLLLLL___]
  // [___________________________SSSSSSSS_]
  // [_________________LLLLLLLL___________]
  // [___________________SSSSSSSS_________]
  // [_________LLLLLLLL___________________]
  // [___________SSSSSSSS_________________]
  // [_____SSSSSSSS_______________________]
  template <typename DstAddrT, typename SrcAddrT>
  static inline void move_backward(DstAddrT dst, SrcAddrT src,
                                   size_t runtime_size) {
    const auto head_value = SizedOpT::load(src);
    ptrdiff_t offset = runtime_size - BLOCK_SIZE;
    do {
      SizedOpT::move(offsetAddrMultiplesOf<BLOCK_SIZE>(dst, offset),
                     offsetAddrMultiplesOf<BLOCK_SIZE>(src, offset));
      offset -= BLOCK_SIZE;
    } while (offset >= 0);
    SizedOpT::store(dst, head_value);
  }

  template <typename DstAddrT>
  static inline void set(DstAddrT dst, ubyte value, size_t runtime_size) {
    size_t offset = 0;
    do {
      SizedOpT::set(offsetAddrMultiplesOf<BLOCK_SIZE>(dst, offset), value);
      offset += BLOCK_SIZE;
    } while (offset < runtime_size - BLOCK_SIZE);
    Tail<SizedOpT>::set(dst, value, runtime_size);
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline uint64_t isDifferent(SrcAddrT1 src1, SrcAddrT2 src2,
                                     size_t runtime_size) {
    size_t offset = 0;
    do {
      if (uint64_t res = SizedOpT::isDifferent(
              offsetAddrMultiplesOf<BLOCK_SIZE>(src1, offset),
              offsetAddrMultiplesOf<BLOCK_SIZE>(src2, offset)))
        return res;
      offset += BLOCK_SIZE;
    } while (offset < runtime_size - BLOCK_SIZE);
    return Tail<SizedOpT>::isDifferent(src1, src2, runtime_size);
  }

  template <typename SrcAddrT1, typename SrcAddrT2>
  static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2,
                                    size_t runtime_size) {
    size_t offset = 0;
    do {
      if (int32_t res = SizedOpT::threeWayCmp(
              offsetAddrMultiplesOf<BLOCK_SIZE>(src1, offset),
              offsetAddrMultiplesOf<BLOCK_SIZE>(src2, offset)))
        return res;
      offset += BLOCK_SIZE;
    } while (offset < runtime_size - BLOCK_SIZE);
    return Tail<SizedOpT>::threeWayCmp(src1, src2, runtime_size);
  }
};

// Aligns using a statically-sized operation, then calls the subsequent NextT
// operation.
//
// e.g. A 16-byte Destination Aligned 32-byte Loop Copy can be written as:
// Align<_16, Arg::Dst>::Then<Loop<_32>>::copy(dst, src, runtime_size);
enum class Arg { _1, _2, Dst = _1, Src = _2, Lhs = _1, Rhs = _2 };
template <typename SizedOpT, Arg AlignOn = Arg::_1> struct Align {
  static_assert(SizedOpT::IS_FIXED_SIZE);

  template <typename NextT> struct Then {
    static_assert(NextT::IS_RUNTIME_SIZE);

    template <typename DstAddrT, typename SrcAddrT>
    static inline void copy(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
      SizedOpT::copy(dst, src);
      auto aligned = align(dst, src, runtime_size);
      NextT::copy(aligned.arg1, aligned.arg2, aligned.size);
    }

    // Move forward suitable when dst < src. The alignment is performed with
    // an HeadTail operation of size ∈ [Alignment, 2 x Alignment].
    //
    // e.g. Moving two bytes and making sure src is then aligned.
    // [  |       |       |       |      ]
    // [____XXXXXXXXXXXXXXXXXXXXXXXXXXXX_]
    // [____LLLLLLLL_____________________]
    // [___________LLLLLLLL______________]
    // [_SSSSSSSS________________________]
    // [________SSSSSSSS_________________]
    //
    // e.g. Moving two bytes and making sure dst is then aligned.
    // [  |       |       |       |      ]
    // [____XXXXXXXXXXXXXXXXXXXXXXXXXXXX_]
    // [____LLLLLLLL_____________________]
    // [______LLLLLLLL___________________]
    // [_SSSSSSSS________________________]
    // [___SSSSSSSS______________________]
    template <typename DstAddrT, typename SrcAddrT>
    static inline void move(DstAddrT dst, SrcAddrT src, size_t runtime_size) {
      auto aligned_after_begin = align(dst, src, runtime_size);
      // We move pointers forward by Size so we can perform HeadTail.
      auto aligned = aligned_after_begin.stepForward();
      HeadTail<SizedOpT>::move(dst, src, runtime_size - aligned.size);
      NextT::move(aligned.arg1, aligned.arg2, aligned.size);
    }

    // Move backward suitable when dst > src. The alignment is performed with
    // an HeadTail operation of size ∈ [Alignment, 2 x Alignment].
    //
    // e.g. Moving two bytes backward and making sure src is then aligned.
    // [  |       |       |       |      ]
    // [____XXXXXXXXXXXXXXXXXXXXXXXX_____]
    // [ _________________LLLLLLLL_______]
    // [ ___________________LLLLLLLL_____]
    // [____________________SSSSSSSS_____]
    // [______________________SSSSSSSS___]
    //
    // e.g. Moving two bytes and making sure dst is then aligned.
    // [  |       |       |       |      ]
    // [____XXXXXXXXXXXXXXXXXXXXXXXX_____]
    // [ _______________LLLLLLLL_________]
    // [ ___________________LLLLLLLL_____]
    // [__________________SSSSSSSS_______]
    // [______________________SSSSSSSS___]
    template <typename DstAddrT, typename SrcAddrT>
    static inline void move_backward(DstAddrT dst, SrcAddrT src,
                                     size_t runtime_size) {
      const auto dst_end = offsetAddrAssumeAligned<1>(dst, runtime_size);
      const auto src_end = offsetAddrAssumeAligned<1>(src, runtime_size);
      auto aligned_after_end = align(dst_end, src_end, 0);
      // We move pointers back by 2 x Size so we can perform HeadTail.
      auto aligned = aligned_after_end.stepBack().stepBack();
      HeadTail<SizedOpT>::move(aligned.arg1, aligned.arg2, aligned.size);
      NextT::move_backward(dst, src, runtime_size - aligned.size);
    }

    template <typename DstAddrT>
    static inline void set(DstAddrT dst, ubyte value, size_t runtime_size) {
      SizedOpT::set(dst, value);
      DstAddrT _(nullptr);
      auto aligned = align(dst, _, runtime_size);
      NextT::set(aligned.arg1, value, aligned.size);
    }

    template <typename SrcAddrT1, typename SrcAddrT2>
    static inline uint64_t isDifferent(SrcAddrT1 src1, SrcAddrT2 src2,
                                       size_t runtime_size) {
      if (const uint64_t res = SizedOpT::isDifferent(src1, src2))
        return res;
      auto aligned = align(src1, src2, runtime_size);
      return NextT::isDifferent(aligned.arg1, aligned.arg2, aligned.size);
    }

    template <typename SrcAddrT1, typename SrcAddrT2>
    static inline int32_t threeWayCmp(SrcAddrT1 src1, SrcAddrT2 src2,
                                      size_t runtime_size) {
      if (const int32_t res = SizedOpT::threeWayCmp(src1, src2))
        return res;
      auto aligned = align(src1, src2, runtime_size);
      return NextT::threeWayCmp(aligned.arg1, aligned.arg2, aligned.size);
    }
  };

private:
  static constexpr size_t ALIGN_OP_SIZE = SizedOpT::SIZE;
  static_assert(ALIGN_OP_SIZE > 1);

  template <typename Arg1AddrT, typename Arg2AddrT> struct Aligned {
    Arg1AddrT arg1;
    Arg2AddrT arg2;
    size_t size;

    Aligned stepForward() const {
      return Aligned{offsetAddrMultiplesOf<ALIGN_OP_SIZE>(arg1, ALIGN_OP_SIZE),
                     offsetAddrMultiplesOf<ALIGN_OP_SIZE>(arg2, ALIGN_OP_SIZE),
                     size - ALIGN_OP_SIZE};
    }

    Aligned stepBack() const {
      return Aligned{offsetAddrMultiplesOf<ALIGN_OP_SIZE>(arg1, -ALIGN_OP_SIZE),
                     offsetAddrMultiplesOf<ALIGN_OP_SIZE>(arg2, -ALIGN_OP_SIZE),
                     size + ALIGN_OP_SIZE};
    }
  };

  template <typename Arg1AddrT, typename Arg2AddrT>
  static auto makeAligned(Arg1AddrT arg1, Arg2AddrT arg2, size_t size) {
    return Aligned<Arg1AddrT, Arg2AddrT>{arg1, arg2, size};
  }

  template <typename Arg1AddrT, typename Arg2AddrT>
  static auto align(Arg1AddrT arg1, Arg2AddrT arg2, size_t runtime_size) {
    static_assert(IsAddressType<Arg1AddrT>::Value);
    static_assert(IsAddressType<Arg2AddrT>::Value);
    if constexpr (AlignOn == Arg::_1) {
      auto offset = offset_to_next_aligned<ALIGN_OP_SIZE>(arg1.ptr_);
      return makeAligned(offsetAddrAssumeAligned<ALIGN_OP_SIZE>(arg1, offset),
                         offsetAddrAssumeAligned<1>(arg2, offset),
                         runtime_size - offset);
    } else if constexpr (AlignOn == Arg::_2) {
      auto offset = offset_to_next_aligned<ALIGN_OP_SIZE>(arg2.ptr_);
      return makeAligned(offsetAddrAssumeAligned<1>(arg1, offset),
                         offsetAddrAssumeAligned<ALIGN_OP_SIZE>(arg2, offset),
                         runtime_size - offset);
    } else {
      DeferredStaticAssert("AlignOn must be either Arg::_1 or Arg::_2");
    }
  }
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ALGORITHM_H
