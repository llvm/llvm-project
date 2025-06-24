#pragma once
#include "rsan_vector.h"

namespace Robustness {
template<
    class Key,
    class T
> class Map {


	Vector<Pair<Key, T>> v;

	u64 findLocationLinear(Key k, u64 start, u64 end){
		for (u64 i=start; i<end;++i)
			if (v[i].first >= k) return i;
		return end;
	}

	u64 find_(Key k, u64 first = 0){
		const auto len = v.size();
		size_t count = len - first;
		while (count > 0) {
			if (count <= 8) return findLocationLinear(k, first, count+first);
			u64 step = count / 2, it = first + step;
			u64 tkey = v[it].first;
			if (tkey > k){
				count = step;
			} else if (tkey < k){
				first = it + 1;
				count -= step + 1;
			} else {
				return it;
			}
		}
		return first;
	}

	public:


	template< class... Args >
	Pair<Pair<Key, T>*, bool> try_emplace( const Key& k, Args&&... args ){
		auto i = find_(k);
		if (i < v.size() && v[i].first == k){
			return pair(&v[i], false);
		} else {
			v.insert(i, pair(k, T(args...)));
			return pair(&v[i], true);
		}
	}

	decltype(v.end()) find(const Key &k){
		auto i = find_(k);
		if (i < v.size() && v[i].first == k)
			return &v[i];
		else
			return v.end();
	}


	decltype(v.begin()) begin(){
		return v.begin();
	}
	decltype(v.begin()) end(){
		return v.end();
	}

	bool contains(const Key &k){
		return find(k) != v.end();
	}

	void clear(){
		v.clear();
	}

	T& operator[]( Key&& key ){
		return this->try_emplace(key).first->second;
	}
	T& operator[]( const Key& key ){
		return this->try_emplace(key).first->second;
	}

	auto size(){
		return v.size();
	}

};
} // namespace Robustness
