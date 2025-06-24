#pragma once
#include "rsan_memoryorder.hpp"
#include "rsan_vectorclock.hpp"
#include "rsan_defs.hpp"
#include "rsan_vector.h"
#include "rsan_map.hpp"
#include "rsan_action.hpp"


namespace Robustness {
	template<class KeyT, class ValueT>
	using map = Robustness::Map<KeyT,ValueT>;

	//! Track SC with VectorClocks
	struct Vsc{
		//! Thread component
		struct Thread{
			VectorClock v, vu;
			//! Absorb other thread into self
			void absorb(const Thread &t){
				v |= t.v;
				vu |= t.vu;
			}
			void resetKnowledge(){
				v.reset();
				vu.reset();
			}
		};
		//! Location component
		struct Location{
			timestamp_t stamp = kEpochZero;
			timestamp_t stampu = kEpochZero;
			VectorClock m, w;
			VectorClock mu, wu;
		};

		/*!
		 * Update load statement
		 *
		 * Memory order is ignored for SC
		 */
		void updateLoadStatement(ThreadId , LocationId , Thread &ts, Location &ls, morder){
			ls.m |= ts.v;
			ts.v |= ls.w;

			ls.mu |= ts.vu;
			ts.vu |= ls.wu;
		}

		/*!
		 * Update store statement
		 *
		 * Memory order is ignored for SC
		 */
		void updateStoreStatement(ThreadId , LocationId a, Thread &ts, Location &ls, morder, u64 val){
			ls.m |= timestamp(a, EpochInc(ls.stamp));
			ts.v |= ls.m;
			ls.w = ts.v;
			ls.m = ts.v;

			ls.mu |= timestamp(a, EpochInc(ls.stampu));
			ts.vu |= ls.mu;
			ls.wu = ts.vu;
			ls.mu = ts.vu;
		}

		/*!
		 * Update RMW statement
		 *
		 * Memory order is ignored for SC
		 */
		void updateRmwStatement(ThreadId t, LocationId a, Thread &ts, Location &ls, morder mo, u64 val){
			//return updateStoreStatement(t, a, ts, ls, mo);
			ls.m |= timestamp(a, EpochInc(ls.stamp));
			ts.v |= ls.m;
			ls.w = ts.v;
			ls.m = ts.v;

			ts.vu |= ls.mu;
			ls.wu = ts.vu;
			ls.mu = ts.vu;
		}

		//! Check if Thread knows of last write to \arg l
		bool knowsLastWrite(ThreadId , LocationId l, Thread &ts, Location &ls) const{
			return ls.w[l] <= ts.v[l];
		}
		timestamp_t getLastTimeStamp(ThreadId , LocationId l, Thread &ts, Location &ls) const{
			return ts.v[l].ts;
		}
		timestamp_t getLastTimeStampU(ThreadId , LocationId l, Thread &ts, Location &ls) const{
			return ts.vu[l].ts;
		}

		//! Remove locations when freeing memory
		void freeLocation(LocationId l, Location &ls){
			ls.w.reset();
			ls.m.reset();
		}
	};


	//! Trace trace with RC20 semantics
	struct VrlxNoFence{
		//! Thread component
		struct Thread{
			VectorClock vc;
			VectorClock vr;
			VectorClock va;
			VectorClock vcu;
			VectorClock vru;
			VectorClock vau;

			//! Absorb thread view into self
			void absorb(const Thread &t){
				vc |= t.vc;
				vr |= t.vr;
				va |= t.va;
				vcu |= t.vcu;
				vru |= t.vru;
				vau |= t.vau;
			}

			void resetKnowledge(ThreadId t){
				vc.reset();
				vr.reset();
				va.reset();
				vcu.reset();
				vru.reset();
				vau.reset();
			}
		};
		//! Location component
		struct Location{
			timestamp_t writeStamp = kEpochZero, writeStampU = kEpochZero;
			VectorClock w;
			VectorClock wu;
		};
		//! Initlialize thread
		void initThread(ThreadId tid, Thread &ts){
		}


		//! Update load statement
		void updateLoadStatement(ThreadId , LocationId a, Thread &ts, Location &ls, morder mo){
			ts.va |= ls.w;
			ts.vau |= ls.wu;
			if (atLeast(mo, (mo_acquire))){
				ts.vc |= ls.w;
				ts.vcu |= ls.wu;
			} else {
				ts.vc |= ls.w[a];
				ts.vcu |= ls.wu[a];
			}
		}

		//! Update store statement
		void updateStoreStatement(ThreadId t, LocationId a, Thread &ts, Location &ls, morder mo, uint64_t oldValue){
			const auto timestampV =  timestamp(a, EpochInc(ls.writeStamp));
			const auto timestampVU = timestamp(a, EpochInc(ls.writeStampU));
			ls.w  |= timestampV;
			ls.wu |= timestampVU;
			ts.va |= timestampV;
			ts.vc |= timestampV;
			ts.vau |= timestampVU;
			ts.vcu |= timestampVU;


			if (atLeast(mo, (mo_release))){
				ls.w = ts.vc;
				ls.wu = ts.vcu;
			} else {
				ls.w = ts.vr;
				ls.w |= timestampV;
				ls.wu = ts.vru;
				ls.wu |= timestampVU;
			}
		}

		//! Update RMW statement
		void updateRmwStatement(ThreadId t, LocationId a, Thread &ts, Location &ls, morder mo, uint64_t oldValue){
			const auto timestampV =  timestamp(a, EpochInc(ls.writeStamp));
			ls.w  |= timestampV;
			ts.va |= timestampV;
			ts.vc |= timestampV;


			ts.va |= ls.w;
			ts.vau |= ls.wu;
			if (atLeast(mo, (mo_acquire))){
				ts.vc |= ls.w;
				ts.vcu |= ls.wu;
			} else {
				ts.vcu |= ls.wu[a];
			}

			if (atLeast(mo, (mo_release))){
				ls.w |= ts.vc;
				ls.wu |= ts.vcu;
			} else {
				ls.w |= ts.vr;
				ls.wu |= ts.vru;
			}
		}


		Mutex SCLock;
		/*!
		 * Update fence statement
		 *
		 * seq_cst fences are compiled to fence(acq); RMW(acq_rel); fence(rel);
		 */
		void updateFenceStatement(ThreadId t, Thread &ts, Location &ls, morder mo){
			if (mo == mo_seq_cst){
				updateFenceStatement(t, ts, ls, mo_acquire);
				{
					Lock instLock(&SCLock);
					updateRmwStatement(t, LocationId(0), ts, ls, mo_acq_rel, 0);
				}
				updateFenceStatement(t, ts, ls, mo_release);
				return;
			}
			if (atLeast(mo, (mo_acquire))){
				ts.vc = ts.va;
				ts.vcu = ts.vau;
			}
			if (atLeast(mo, (mo_release))){
				ts.vr = ts.vc;
				ts.vru = ts.vcu;
			}
		}

		auto getLastTimeStamp(ThreadId t, LocationId l, Thread &ts, Location &ls){
			return ts.vc[l].ts;
		}
		auto getLastTimeStampU(ThreadId t, LocationId l, Thread &ts, Location &ls){
			return ts.vcu[l].ts;
		}



		//! Remove locations when freeing memory
		void freeLocation(LocationId l, Location &ls){
			ls.w.reset();
			ls.wu.reset();
		}
	};


	/// Instrumentation
class Instrumentation{
	public:
	virtual void verifyLoadStatement(ThreadId t, LocationId l, morder mo) = 0;
	virtual void verifyStoreStatement(ThreadId t, LocationId l, morder mo) = 0;
	virtual void updateLoadStatement(Action::AtomicLoadAction a) = 0;
	virtual void updateStoreStatement(Action::AtomicStoreAction a) = 0;
	virtual void updateRmwStatement(Action::AtomicRMWAction a) = 0;
	virtual void updateCasStatement(Action::AtomicCasAction a) = 0;
	virtual void updateFenceStatement(ThreadId t, morder mo) = 0;
	virtual void updateNALoad(ThreadId t, LocationId l) = 0;
	virtual void updateNAStore(ThreadId t, LocationId l) = 0;

	virtual void absorbThread(ThreadId _absorber, ThreadId _absorbee) = 0;
	virtual void cloneThread(ThreadId _src, ThreadId dst) = 0;
	//void initThread(ThreadId tid);
	//void removeThread(ThreadId t);

	virtual int64_t getViolationsCount() = 0;
	virtual int64_t getRacesCount() = 0;
	virtual void freeMemory(ThreadId t, LocationId l, ptrdiff_t size) = 0;

	virtual void trackAtomic(ThreadId t, LocationId l, uint64_t val) = 0;
	virtual void waitAtomic(Action::WaitAction a) = 0;
	virtual void bcasAtomic(Action::BcasAction a) = 0;

	virtual ~Instrumentation() = default;
};

} // namespace Robustness

