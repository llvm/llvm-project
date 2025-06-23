#pragma once
#include "tsan_defs.h"
#include "tsan_interface.h"
namespace Robustness{
	using __tsan::morder;
	using __tsan::mo_relaxed;
	using __tsan::mo_consume;
	using __tsan::mo_acquire;
	using __tsan::mo_release;
	using __tsan::mo_acq_rel;
	using __tsan::mo_seq_cst;
	//! Check if lhs is at least as strong as rhs.
	/*!
	 * Check if memory order is at least as strong as another
	 * \param lhs memory order
	 * \param rhs memory order
	 * \return true if lhs is at least as powerful as rhs
	 */
	inline bool atLeast(__tsan::morder lhs, __tsan::morder rhs){
		using namespace std;
		switch (rhs) {
			case __tsan::mo_relaxed:
				return true;
			case __tsan::mo_consume:
			case __tsan::mo_acquire:
				switch (lhs) {
					case __tsan::mo_relaxed:
					case __tsan::mo_release:
						return false;
					case __tsan::mo_acq_rel:
					case __tsan::mo_acquire:
					case __tsan::mo_seq_cst:
						return true;
					case __tsan::mo_consume:
						//assertm("Consume not supported", 0);
					default:
						//assertm("Unknown memory order value", 0);
						// TODO: Remove bugs from here
						return false;
				}
			case __tsan::mo_release:
				switch (lhs) {
					case __tsan::mo_relaxed:
					case __tsan::mo_acquire:
					case __tsan::mo_consume:
						return false;
					case __tsan::mo_acq_rel:
					case __tsan::mo_release:
					case __tsan::mo_seq_cst:
						return true;
					default:
						// TODO: Remove bugs from here
						//assertm("Unknown memory order value", 0);
						return false;
				}
			case __tsan::mo_acq_rel:
				return lhs == __tsan::mo_seq_cst || lhs == __tsan::mo_acq_rel;
			case __tsan::mo_seq_cst:
				return lhs == __tsan::mo_seq_cst;
		}
		//assertm(0, "Unhandeled atLeast for some memory order");
		__builtin_unreachable();
	}
} // namespace Robustness

