#pragma once
#include "rsan_defs.hpp"
#include "rsan_vector.h"

namespace Robustness{

template <typename T> class MiniMapClock;

/**
 * Timestamp
 */
template <typename T>
class Timestamp{
	public:
	T key{};
	timestamp_t ts = kEpochZero;

	/// Check if the timestamp is newer than rhs
	public:
	bool contains(Timestamp<T> rhs) const{
		return key == rhs.key && ts >= rhs.ts;
	}
	//auto operator<=>(const Timestamp<T>&) const = default;
};
	template <typename T> inline bool operator< (const Timestamp<T>& lhs, const Timestamp<T>& rhs) { return lhs.key < rhs.key ? true : lhs.key == rhs.key ? lhs.ts < rhs.ts : false; }
	template <typename T> inline bool operator==(const Timestamp<T>& lhs, const Timestamp<T>& rhs) { return lhs.key == rhs.key && lhs.ts == rhs.ts; }
	template <typename T> inline bool operator> (const Timestamp<T>& lhs, const Timestamp<T>& rhs) { return rhs < lhs; }
	template <typename T> inline bool operator<=(const Timestamp<T>& lhs, const Timestamp<T>& rhs) { return !(lhs > rhs); }
	template <typename T> inline bool operator>=(const Timestamp<T>& lhs, const Timestamp<T>& rhs) { return !(lhs < rhs); }
	template <typename T> inline bool operator!=(const Timestamp<T>& lhs, const Timestamp<T>& rhs) { return !(lhs == rhs); }

template <typename T>
auto timestamp(T key, timestamp_t ts){
	return Timestamp<T>{key, ts};
}

/**
  Vector Clock
  **/

template<class T> struct remove_reference { typedef T type; };
template<class T> struct remove_reference<T&> { typedef T type; };
template<class T> struct remove_reference<T&&> { typedef T type; };

class VectorClock {
	private:
		Robustness::Vector<timestamp_t> impl;

	public:
		/// Increment a timestamp t in the vector
		auto inc(LocationId t){
			impl.ensureSize(t+1);
			timestamp(t, EpochInc(impl[t]));
		}
		/// Reset the vector clock
		void reset(){
			impl.clear();
		}
		void receive(Timestamp<LocationId> ts){
			const auto loc = ts.key;
			impl.ensureSize(loc+1);
			impl[loc] = max(impl[loc], ts.ts);
		}

		/**
		  Support
		  |= Union
		 **/
		VectorClock& operator|=(const VectorClock &rhs){
			auto S1 = impl.size();
			auto S2 = rhs.impl.size();
			impl.ensureSize(S2);
			auto S = min(S1,S2);
			uptr i = 0;
			for (i = 0; i < S; ++i){
				impl[i] = max(impl[i], rhs.impl[i]);
			}
			for (i = S; i < S2; ++i){
				impl[i] = rhs.impl[i];
			}
			return *this;
		}


		/**
		  |= - add a timestamp
		 **/
		auto& operator|=(const Timestamp<LocationId> &rhs){
			receive(rhs);
			return *this;
		}
		bool contains(const VectorClock &rhs) const{
			auto S1 = impl.size(), S2 = rhs.impl.size();
			decltype(S1) i = 0;
			for (; i < S1 && i < S2; ++i)
				if (impl[i] < rhs.impl[i])
					return false;
			for (; i < S2; ++i)
				if (rhs.impl[i] > kEpochZero)
					return false;
			return true;
		}

		auto operator[](LocationId t) const {
			if (t < impl.size()) {
				return timestamp(t, impl[t]);
			}
			return timestamp(t, kEpochZero);
		}

		bool contains(const Timestamp<LocationId> &rhs) const{
			return operator[](rhs.key) >= rhs;
		}
};
} // namespace Robustness
