#pragma once
#include "rsan_vector.h"
#include "rsan_defs.hpp"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace Robustness {
	template< class T >
		class Arena {

			//const FACTOR = 2;
			static const u8 BASE = 8;

			u64 cv = 0;
			u64 ci = 0;

			Vector<Vector<T>> vs;
			Arena(const Arena&) = delete;


			public:
			Arena() = default;
			~Arena() {
				for (auto& v : vs)
					v.clear();
			}

			T* allocate(){
				if (cv == vs.size()){
					vs.push_back();
					vs[cv].resize(BASE << (cv));
					ci = 0;
				}
				DCHECK_GT(vs.size(), cv);
				DCHECK_GT(vs[cv].size(), ci);
				auto ret = &vs[cv][ci++];
				DCHECK_GT(ret, 0);
				if (ci >= vs[cv].size()){
					++cv;
				}

				new (ret) T();
				return ret;
			}
		};
} // namespace Robustness
