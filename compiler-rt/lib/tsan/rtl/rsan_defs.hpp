#pragma once
#include "tsan_defs.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"

//class __tsan::ThreadState;

namespace Robustness{
	using __tsan::s8;
	using __tsan::u8;
	using __tsan::s16;
	using __tsan::u16;
	using __tsan::s32;
	using __tsan::u32;
	using __tsan::s64;
	using __tsan::u64;
	using __tsan::uptr;
	using __tsan::Epoch;
	using __tsan::EpochInc;
	using __tsan::EpochOverflow;
	using __tsan::kEpochZero;
	using __tsan::kEpochOver;
	using __tsan::kEpochLast;
	typedef __tsan::Epoch timestamp_t;
	typedef s64 ssize_t;
	typedef u64 uint64_t;
	typedef s64 int64_t;
	typedef __PTRDIFF_TYPE__ ptrdiff_t;
	typedef __SIZE_TYPE__ size_t;

	typedef u8 uint8_t;;

	typedef u64 Address;
	typedef u64 LocationId;
	
	typedef u32 ThreadId;

	using __tsan::InternalScopedString;

	using __tsan::flags;

	using __sanitizer::IsAligned;

	using __sanitizer::LowLevelAllocator;
	using __sanitizer::InternalAlloc;
	using __sanitizer::InternalFree;
	using __sanitizer::internal_memcpy;
	using __sanitizer::internal_memmove;
	using __sanitizer::internal_memset;
	using __sanitizer::RoundUpTo;
	using __sanitizer::RoundUpToPowerOfTwo;
	using __sanitizer::GetPageSizeCached;
	using __sanitizer::MostSignificantSetBitIndex;
	using __sanitizer::MmapOrDie;
	using __sanitizer::UnmapOrDie;
	using __sanitizer::Max;
	using __sanitizer::Swap;
	using __sanitizer::forward;
	using __sanitizer::move;

	using __sanitizer::Printf;
	using __sanitizer::Report;

	using __sanitizer::Lock;
	using __sanitizer::Mutex;

	template <typename T1, typename T2>
	struct Pair{
		T1 first;
		T2 second;
	};
	template <typename T1, typename T2>
	auto pair(T1 fst, T2 snd){
		return Pair<T1, T2>{fst, snd};
	}

	using __tsan::max;
	using __tsan::min;

	enum class ViolationType{
		read, write,
	};

	struct DebugInfo {
		__tsan::ThreadState* thr = nullptr;
		uptr pc = 0xDEADBEEF;
	};

	template<class>
		inline constexpr bool always_false_v = false;

	inline bool isRobustness() {
		return __tsan::flags()->enable_robustness;
	}
} //  namespace Robustness
