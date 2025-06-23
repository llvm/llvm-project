#pragma once
#include "rsan_defs.hpp"
namespace Robustness::Action{
	struct StoreAction{
		ThreadId tid;
		Address addr;
		int size;
	};
	struct LoadAction{
		ThreadId tid;
		Address addr;
		int size;
	};
	struct AtomicVerifyAction{
		ThreadId tid;
		Address addr;
		morder mo;
		int size;
	};
	struct AtomicVerifyStoreAction{
		ThreadId tid;
		Address addr;
		morder mo;
		int size;
	};
	struct AtomicLoadAction{
		ThreadId tid;
		Address addr;
		morder mo;
		int size;
		bool rmw;
		DebugInfo dbg;
	};
	struct AtomicStoreAction{
		ThreadId tid;
		Address addr;
		morder mo;
		int size;
		uint64_t oldValue;
		uint64_t newValue;
		DebugInfo dbg;
	};
	struct AtomicRMWAction{
		ThreadId tid;
		Address addr;
		morder mo;
		int size;
		uint64_t oldValue;
		uint64_t newValue;
		DebugInfo dbg;
	};
	struct AtomicCasAction{
		ThreadId tid;
		Address addr;
		morder mo;
		int size;
		uint64_t oldValue;
		uint64_t newValue;
		bool success;
		DebugInfo dbg;
	};
	struct FenceAction{
		ThreadId tid;
		morder mo;
	};
	struct TrackAction{
		ThreadId tid;
		Address addr;
		uint64_t value;
	};
	struct WaitAction{
		ThreadId tid;
		Address addr;
		uint64_t value;
		DebugInfo dbg;
	};
	struct BcasAction{
		ThreadId tid;
		Address addr;
		uint64_t value;
		DebugInfo dbg;
	};
	struct ThreadCreate{
		ThreadId creator, createe;
	};
	struct ThreadJoin{
		ThreadId absorber, absorbee;
	};
	struct Free{
		ThreadId tid;
		Address addr;
		uptr size;
		DebugInfo dbg;
	};
}


