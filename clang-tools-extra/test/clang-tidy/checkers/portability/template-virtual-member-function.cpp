// RUN: %check_clang_tidy %s portability-template-virtual-member-function %t
namespace UninstantiatedVirtualMember {
template<typename T>
struct CrossPlatformError {
    virtual ~CrossPlatformError() = default;

    static void used() {}

    // CHECK-MESSAGES: [[#@LINE+1]]:18: warning: unspecified virtual member function instantiation
    virtual void unused() {
        T MSVCError = this;
    };
};

int main() {
    // CHECK-MESSAGES: [[#@LINE+1]]:5: note: template instantiated here
    CrossPlatformError<int>::used();
    return 0;
}
} // namespace UninstantiatedVirtualMember

namespace UninstantiatedVirtualMembers {
template<typename T>
struct CrossPlatformError {
    virtual ~CrossPlatformError() = default;

    static void used() {}

    // CHECK-MESSAGES: [[#@LINE+2]]:18: warning: unspecified virtual member function instantiation
    // CHECK-MESSAGES: [[#@LINE+13]]:5: note: template instantiated here
    virtual void unused() {
        T MSVCError = this;
    };

    // CHECK-MESSAGES: [[#@LINE+2]]:18: warning: unspecified virtual member function instantiation
    // CHECK-MESSAGES: [[#@LINE+7]]:5: note: template instantiated here
    virtual void unused2() {
        T MSVCError = this;
    };
};

int main() {
    CrossPlatformError<int>::used();
    return 0;
}
} // namespace UninstantiatedVirtualMembers

namespace UninstantiatedVirtualDestructor {
template<typename T>
struct CrossPlatformError {
    // CHECK-MESSAGES: [[#@LINE+2]]:13: warning: unspecified virtual member function instantiation
    // CHECK-MESSAGES: [[#@LINE+9]]:5: note: template instantiated here
    virtual ~CrossPlatformError() {
        T MSVCError = this;
    };

    static void used() {}
};

int main() {
    CrossPlatformError<int>::used();
    return 0;
}
} // namespace UninstantiatedVirtualDestructor

namespace MultipleImplicitInstantiations {
template<typename T>
struct CrossPlatformError {
    virtual ~CrossPlatformError() = default;

    static void used() {}

    // CHECK-MESSAGES: [[#@LINE+2]]:18: warning: unspecified virtual member function instantiation
    // CHECK-MESSAGES: [[#@LINE+7]]:5: note: template instantiated here
    virtual void unused() {
        T MSVCError = this;
    };
};

int main() {
    CrossPlatformError<int>::used();
    CrossPlatformError<float>::used();
    CrossPlatformError<long>::used();
    return 0;
}
} // namespace MultipleImplicitInstantiations

namespace SomeImplicitInstantiationError {
template <typename T> struct CrossPlatformError {
    virtual ~CrossPlatformError() = default;

    static void used() {}

    // CHECK-MESSAGES: [[#@LINE+2]]:18: warning: unspecified virtual member function instantiation
    // CHECK-MESSAGES: [[#@LINE+5]]:5: note: template instantiated here
    virtual void unused(){};
};

int main() {
    CrossPlatformError<int>::used();
    CrossPlatformError<float> NoError;
    return 0;
}
} // namespace SomeImplicitInstantiationError

namespace InstantiatedVirtualMemberFunctions {
template<typename T>
struct NoError {
    virtual ~NoError() {};
    virtual void unused() {};
    virtual void unused2() {};
    virtual void unused3() {};
};

int main() {
    NoError<int> Ne;
    return 0;
}
} // namespace InstantiatedVirtualMemberFunctions

namespace UninstantiatedNonVirtualMemberFunctions {
template<typename T>
struct NoError {
    static void used() {};
    void unused() {};
    void unused2() {};
    void unused3() {};
};

int main() {
    NoError<int>::used();
    return 0;
}
} // namespace UninstantiatedNonVirtualMemberFunctions

namespace PartialSpecializationError {
template<typename T, typename U>
struct CrossPlatformError {};

template<typename U>
struct CrossPlatformError<int, U>{
    virtual ~CrossPlatformError() = default;

    static void used() {}

    // CHECK-MESSAGES: [[#@LINE+2]]:18: warning: unspecified virtual member function instantiation
    // CHECK-MESSAGES: [[#@LINE+7]]:5: note: template instantiated here
    virtual void unused() {
        U MSVCError = this;
    };
};

int main() {
    CrossPlatformError<int, float>::used();
    return 0;
}
} // namespace PartialSpecializationError

namespace PartialSpecializationNoInstantiation {
template<typename T, typename U>
struct NoInstantiation {};

template<typename U>
struct NoInstantiation<int, U>{
    virtual ~NoInstantiation() = default;

    static void used() {}

    virtual void unused() {
        U MSVCError = this;
    };
};
} // namespace PartialSpecializationNoInstantiation
