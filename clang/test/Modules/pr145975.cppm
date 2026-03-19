// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm \
// RUN:     -fmodule-file=a:b=%t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -o %t/c.pcm \
// RUN:     -fmodule-file=a:b=%t/b.pcm -fmodule-file=a=%t/a.pcm

// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm \
// RUN:     -fmodule-file=a:b=%t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-reduced-module-interface -o %t/c.pcm \
// RUN:     -fmodule-file=a:b=%t/b.pcm -fmodule-file=a=%t/a.pcm


//--- b.cppm
export module a:b;
class Polymorphic {
public:
    virtual ~Polymorphic() = default;
};

//--- a.h
namespace std {
using X = int;
}

namespace a {
    using std::X;
}

//--- a.cppm
module;
#include "a.h"
export module a;
import :b;
std::X var;
namespace std {
    export using std::X;
}
namespace a {
    export using std::X;
}

//--- c.cppm
export module c;
import a;
namespace a {
    export using std::X;
}

namespace a {
    X test() {
        return X{};
    }
}
