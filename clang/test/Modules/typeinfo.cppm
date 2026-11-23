// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/typeinfo.cppm -emit-module-interface -o %t/typeinfo.pcm
// RUN: %clang_cc1 -std=c++20 %t/typeid.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/typeinfo.cppm -emit-reduced-module-interface -o %t/typeinfo.pcm
// RUN: %clang_cc1 -std=c++20 %t/typeid.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- typeinfo.h
namespace std {
class type_info {
public:
    virtual ~type_info();
    const char* name() const { return __name; }
    bool operator==(const type_info& __arg) const {
    return __name == __arg.__name;
    }

    bool operator!=(const type_info& __arg) const {
    return !operator==(__arg);
    }

    bool before(const type_info& __arg) const {
    return __name < __arg.__name;
    }

    unsigned long hash_code() const {
    return reinterpret_cast<unsigned long long>(__name);
    }
protected:
    const char *__name;
};
}

//--- typeinfo.cppm
module;
#include "typeinfo.h";
export module typeinfo;
export namespace std {
    using std::type_info;
}

//--- typeid.cpp
// expected-no-diagnostics
import typeinfo;

class A {
public:
    virtual void foo();
};

class B : public A {
public:
    void foo() override;
};

void A::foo() {}
void B::foo() {}

const auto &getTypeInfo() {
    return typeid(A);
}

const char *getName() {
    return typeid(A).name();
}

bool equal(A *a) {
    return typeid(B) == typeid(*a);
}
