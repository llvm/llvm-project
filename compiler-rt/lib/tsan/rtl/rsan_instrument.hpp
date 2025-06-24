#pragma once
#include "rsan_robustnessmodel.hpp"
#include "rsan_map.hpp"
#include "rsan_defs.hpp"
#include "rsan_report.hpp"
#include "rsan_stacktrace.hpp"
#include "rsan_arena.hpp"
#include "rsan_lock.hpp"

namespace Robustness{

	static FakeMutex fakeMutex;
	/*!
	 * Insrumentation
	 */
	template <typename I>
		//struct InstrumentationTemplate : Instrumentation{
		struct InstrumentationTemplate {
			private:
				Vsc vsc; //!< VSC tracking
				I ins; //!< Memory Model tracking
				int64_t violationsCount = 0;

				Mutex locksLock; // Global Robustness Lock
				Arena<Mutex> locksAllocator;


				template <typename KeyT, typename ValueT>
					using map = Robustness::Map<KeyT,ValueT>;

				map<Address, Mutex*> locks;


				//! Thread part
				struct ThreadStruct{
					Vsc::Thread vsc;
					typename I::Thread ins;
					/*! Absorb another thread
					 * \param w Thread struct to absorb
					 */
					void absorb(const ThreadStruct &w){
						vsc.absorb(w.vsc);
						ins.absorb(w.ins);
					}
					void resetKnowledge(const ThreadId &t){
						vsc.resetKnowledge();
						ins.resetKnowledge(t);
					}
				};
				//! Location Part
				struct LocationStruct{
					Vsc::Location vsc;
					typename I::Location ins;
					LittleStackTrace lastWrite;
					LittleStackTrace lastWriteU;
					LocationId lid;
				};
				//volatile LocationId locationCounter{1};
				u64 locationCounter{0};
				// Location 0 is reserved for SC fences

				Mutex structsLock; // Global Robustness Lock
				Arena<ThreadStruct> threadAllocator;
				Arena<LocationStruct> locationAllocator;

				map<ThreadId, ThreadStruct*> threads;
				map<Address, LocationStruct*> locations;

				/*!
				 * Get Location Struct for address
				 */
				inline auto& getLocationStruct(Address a) {
					Lock lock(&structsLock);
					if (auto it = locations.find(a); it != locations.end()){
						return *it->second;
					}
					auto w = locationAllocator.allocate();
					//w->lid =  __atomic_fetch_add(&locationCounter, 1, __ATOMIC_SEQ_CST) ;
					w->lid = ++locationCounter;
					locations[a] = w;
					return *w;
				}
				/*!
				 * Get Thread Struct for address
				 */
				inline auto& getThreadStruct(ThreadId tid) {
					Lock lock(&structsLock);
					if (auto it = threads.find(tid); it != threads.end()){
						return *it->second;
					}
					auto w = threadAllocator.allocate();
					threads[tid] = w;
					return *w;
				}
				/*!
				 * Get Location Struct for address only if exists
				 */
				inline auto getMaybeLocationStruct(Address a) {
					Lock lock(&structsLock);
					auto t = locations.find(a);
					return (t != locations.end() ? t->second : nullptr);
				}


				//! returns the number of violations
				virtual int64_t getViolationsCount() /*override*/{
					return violationsCount;
				}

				/*!
				 * Assert no read violation occurs
				 *
				 * \param t Thread Id
				 * \param l Address
				 * \param ts ThreadStruct
				 * \param ls Location Struct
				 */
				void assertReadViolation(ThreadId t, Address a, ThreadStruct &ts, LocationStruct &ls, DebugInfo dbg) {
					auto l = ls.lid;
					if (vsc.getLastTimeStamp(t, l, ts.vsc, ls.vsc) > ins.getLastTimeStamp(t, l, ts.ins, ls.ins)){
						reportViolation<ViolationType::read>(t, a, l, dbg, ls.lastWrite);
					}
				}
				/*!
				 * Assert no write violation occurs
				 *
				 * \param t Thread Id
				 * \param l Address
				 * \param ts ThreadStruct
				 * \param ls Location Struct
				 */
				void assertWriteViolation(ThreadId t, Address a, ThreadStruct &ts, LocationStruct &ls, DebugInfo dbg) {
					auto l = ls.lid;
					if (vsc.getLastTimeStampU(t, l, ts.vsc, ls.vsc) > ins.getLastTimeStampU(t, l, ts.ins, ls.ins)){
						reportViolation<ViolationType::write>(t, a, l, dbg, ls.lastWriteU);
					}
				}

				void assertCasViolation(ThreadId t, Address a, ThreadStruct &ts, LocationStruct &ls, DebugInfo dbg, uint64_t val) {
					// Weak CAS
					assertReadViolation(t, a, ts, ls, dbg);
				}

				void assertStrongCasViolation(ThreadId t, Address a, ThreadStruct &ts, LocationStruct &ls, DebugInfo dbg, uint64_t val) {
					//auto l = ls.lid;
					//if (vsc.getLastTimeStampUV(t, l, ts.vsc, ls.vsc, val) >= ins.getLastTimeStampU(t, l, ts.ins, ls.ins)){
					//	reportViolation<ViolationType::write>(t, a, l, dbg, ls.lastWriteU);
					//} else if (vsc.getLastTimeStamp(t, l, ts.vsc, ls.vsc, val) >= ins.getLastTimeStamp(t, l, ts.ins, ls.ins)){
					//		reportViolation<ViolationType::read>(t, a, l, dbg, ls.lastWrite);
					//}
				}


			public:
				/*!
				 * Verify load statement for violation without updating
				 *
				 * \param t tid
				 * \param l address
				 */
				void verifyLoadStatement(ThreadId t, Address addr, morder , DebugInfo dbg) /*override*/{
					auto &ls = getLocationStruct(addr);
					auto &ts = getThreadStruct(t);
					assertReadViolation(t, addr, ts, ls, dbg);
				}
				/*!
				 * Verify store statement for violation without updating
				 *
				 * \param t tid
				 * \param l address
				 */
				void verifyStoreStatement(ThreadId t, Address addr, morder , DebugInfo dbg) /*override*/{
					auto &ls = getLocationStruct(addr);
					auto &ts = getThreadStruct(t);
					assertWriteViolation(t, addr, ts, ls, dbg);
				}

				/*!
				 * Verify robustness and update load statement
				 *
				 * \param t tid
				 * \param l address
				 * \param mo memory order
				 */
				void updateLoadStatement(Action::AtomicLoadAction a) /*override*/{
					ThreadId t = a.tid;
					Address addr = a.addr;
					morder mo = a.mo;
					auto &ls = getLocationStruct(addr);
					auto &ts = getThreadStruct(t);
					LocationId l = ls.lid;
					assertReadViolation(t, addr, ts, ls, a.dbg);

					vsc.updateLoadStatement(t, l, ts.vsc, ls.vsc, mo);
					ins.updateLoadStatement(t, l, ts.ins, ls.ins, mo);
				}

				/*!
				 * Verify robustness and update store statement
				 *
				 * \param t tid
				 * \param l address
				 * \param mo memory order
				 */
				void updateStoreStatement(Action::AtomicStoreAction a) /*override*/{
					ThreadId t = a.tid;
					Address addr = a.addr;
					morder mo = a.mo;
					uint64_t val = a.oldValue;
					auto &ls = getLocationStruct(addr);
					auto &ts = getThreadStruct(t);
					LocationId l = ls.lid;
					assertWriteViolation(t, addr, ts, ls, a.dbg);


					vsc.updateStoreStatement(t, l, ts.vsc, ls.vsc, mo, val);
					ins.updateStoreStatement(t, l, ts.ins, ls.ins, mo, val);


					ObtainCurrentLine(a.dbg.thr, a.dbg.pc, &ls.lastWrite);
					ObtainCurrentLine(a.dbg.thr, a.dbg.pc, &ls.lastWriteU);
				}

				/*!
				 * Verify robustness and update RMW statement
				 *
				 * \param t tid
				 * \param l address
				 * \param mo memory order
				 */
				void updateRmwStatement(Action::AtomicRMWAction a) /*override*/{
					ThreadId t = a.tid;
					Address addr = a.addr;
					morder mo = a.mo;
					uint64_t val = a.oldValue;
					auto &ls = getLocationStruct(addr);
					auto &ts = getThreadStruct(t);
					LocationId l = ls.lid;
					assertWriteViolation(t, addr, ts, ls, a.dbg);

					vsc.updateRmwStatement(t, l, ts.vsc, ls.vsc, mo, val);
					ins.updateRmwStatement(t, l, ts.ins, ls.ins, mo, val);

					ObtainCurrentLine(a.dbg.thr, a.dbg.pc, &ls.lastWrite);
				}

				void updateCasStatement(Action::AtomicCasAction a) /*override*/{
					ThreadId t = a.tid;
					Address addr = a.addr;
					morder mo = a.mo;
					uint64_t expected = a.oldValue;
					bool success = a.success;
					auto &ls = getLocationStruct(addr);
					auto &ts = getThreadStruct(t);
					LocationId l = ls.lid;
					assertCasViolation(t, addr, ts, ls, a.dbg, expected);

					if (success){
						vsc.updateRmwStatement(t, l, ts.vsc, ls.vsc, mo, expected);
						ins.updateRmwStatement(t, l, ts.ins, ls.ins, mo, expected);
						ObtainCurrentLine(a.dbg.thr, a.dbg.pc, &ls.lastWrite);
					} else {
						vsc.updateLoadStatement(t, l, ts.vsc, ls.vsc, mo);
						ins.updateLoadStatement(t, l, ts.ins, ls.ins, mo);
					}

				}

				/*!
				 * Update fence statement
				 *
				 * \param t tid
				 * \param mo memory order
				 */
				void updateFenceStatement(ThreadId t, morder mo) /*override*/{
					// HACK: This might break on architectures that use the address 0
					auto &ls = getLocationStruct(0);
					auto &ts = getThreadStruct(t);
					ins.updateFenceStatement(t, ts.ins, ls.ins, mo);
				}

				/*!
				 * Absorb knowledge from thread (Join)
				 *
				 * \param _absorber The thread to gain the knowledge
				 * \param _absorbee The thread giving the knowldge
				 */
				void absorbThread(ThreadId _absorber, ThreadId _absorbee) /*override*/{
					auto &absorber = getThreadStruct(_absorber);
					auto &absorbee = getThreadStruct(_absorbee);
					absorber.absorb(absorbee);
				}

				//! Initialize thread data structure
				void initThread(ThreadId tid) {
					ins.initThread(tid, getThreadStruct(tid).ins);
				}

				/*!
				 * Clone knowledge of current thread to a new thread
				 *
				 * \param _src The thread creating the new thread
				 * \param _dst The newely created thread
				 */
				void cloneThread(ThreadId _src, ThreadId _dst)  /*override*/{
					auto &dst = getThreadStruct(_dst);
					auto &src = getThreadStruct(_src);
					dst.resetKnowledge(_dst);
					dst.absorb(src);
					initThread(_dst);
				}


				/*!
				 * Free chunk of memory, removing knowledge by all relations and
				 * verifying the deletion doesn't violate anything
				 *
				 * \param t tid
				 * \param l Address
				 * \param size size from address
				 */
				void freeMemory(Action::Free w) /*override*/{
					auto &t = w.tid;
					auto &addr = w.addr;
					auto &size = w.size;

					auto &ts = getThreadStruct(t);


					// We don't free the memory. We just mark the location as known to all.
					for (auto a = addr; a <addr+size; ++a){
						if (a == 0) continue;
						auto ls = getMaybeLocationStruct(a);
						if (ls){
							Lock instLock(getLockForAddr(a));
							assertWriteViolation(t, a, ts, *ls, w.dbg);
							vsc.freeLocation(ls->lid, ls->vsc);
							ins.freeLocation(ls->lid, ls->ins);
						}
					}
				}

				Mutex* getLockForAddr(Address addr){
					if (!Robustness::isRobustness())
						return &fakeMutex;
					Lock lock(&locksLock);
					if (auto it = locks.find(addr); it != locks.end()){
						return it->second;
					}
					auto newLock = locksAllocator.allocate();
					locks[addr] = newLock;
					return newLock;
				}

		};

	inline Robustness::InstrumentationTemplate<Robustness::VrlxNoFence> ins;
} // namespace Robustness
