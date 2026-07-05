// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 -Wno-experimental-header-units -fmodule-header %t/RelaxedAtomic.h -o %t/RelaxedAtomic.pcm
// RUN: %clang -std=c++20 -Wno-experimental-header-units -fmodule-header -fmodule-file=%t/RelaxedAtomic.pcm %t/SharedMutex.h -o %t/SharedMutex.pcm
// RUN: %clang -std=c++20 -Wno-experimental-header-units -fmodule-header -fmodule-file=%t/SharedMutex.pcm -fmodule-file=%t/RelaxedAtomic.pcm %t/ThreadLocalDetail.h -o %t/ThreadLocalDetail.pcm
//--- RelaxedAtomic.h
struct relaxed_atomic_base {
  relaxed_atomic_base(int) {}
};

struct relaxed_atomic : relaxed_atomic_base {
  using relaxed_atomic_base::relaxed_atomic_base; // constructor
};

//--- SharedMutex.h
import "RelaxedAtomic.h";

inline void getMaxDeferredReaders() {
  static relaxed_atomic cache{0};
}

//--- ThreadLocalDetail.h
import "RelaxedAtomic.h";

struct noncopyable {
  noncopyable(const noncopyable&) = delete;
};

struct StaticMetaBase {
  relaxed_atomic nextId_{0};
  noncopyable ncp;
};
