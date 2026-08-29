// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

class PublicBase {
    public:
    PublicBase(int i);
    using type = int;
};

class ProtectedBase {
    protected:
    ProtectedBase(int i);
    using type = int;
};

class FriendProtectedBase {
    template <typename U> friend class Derived;

    protected:
    FriendProtectedBase(int i);
    using type = int;
};

class PrivateBase {
    private:
    PrivateBase(int i);
    using type = int;
};

template <typename... Ts>
using void_t = void;

template <typename U>
class Derived {
    public:
    // SFINAE Pattern 1 - Checks whether constructor is accessible
    template <typename T, typename = void>
    struct ConstructsBase {
        static constexpr int value = 0;
    };

    template <typename T>
    struct ConstructsBase<
        T, void_t<decltype(T(0))>> {
            static constexpr int value = 1;
        };

    // SFINAE Pattern 2 - Checks whether type alias is accessible
    template <typename T, typename = void>
    struct HasAccessibleType {
        static constexpr int value = 0;
    };

    template <typename T>
    struct HasAccessibleType<
        T,
        void_t<typename T::type>> 
    {
        static constexpr int value = 1;
    };
};

static_assert(Derived<int>::ConstructsBase<PublicBase>::value);
static_assert(!Derived<int>::ConstructsBase<ProtectedBase>::value);
static_assert(Derived<int>::ConstructsBase<FriendProtectedBase>::value);
static_assert(!Derived<int>::ConstructsBase<PrivateBase>::value);

static_assert(Derived<int>::HasAccessibleType<PublicBase>::value);
static_assert(!Derived<int>::HasAccessibleType<ProtectedBase>::value);
static_assert(Derived<int>::HasAccessibleType<FriendProtectedBase>::value);
static_assert(!Derived<int>::HasAccessibleType<PrivateBase>::value);