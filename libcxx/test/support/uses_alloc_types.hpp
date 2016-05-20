//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef USES_ALLOC_TYPES_HPP
#define USES_ALLOC_TYPES_HPP

# include <experimental/memory_resource>
# include <experimental/utility>
# include <memory>
# include <cassert>

#include "test_memory_resource.hpp"
#include "type_id.h"

// There are two forms of uses-allocator construction:
//   (1) UA_AllocArg: 'T(allocator_arg_t, Alloc const&, Args&&...)'
//   (2) UA_AllocLast: 'T(Args&&..., Alloc const&)'
// 'UA_None' represents non-uses allocator construction.
enum class UsesAllocatorType {
  UA_None = 0,
  UA_AllocArg = 2,
  UA_AllocLast = 4
};
constexpr UsesAllocatorType UA_None = UsesAllocatorType::UA_None;
constexpr UsesAllocatorType UA_AllocArg = UsesAllocatorType::UA_AllocArg;
constexpr UsesAllocatorType UA_AllocLast = UsesAllocatorType::UA_AllocLast;

template <class Alloc, std::size_t N>
class UsesAllocatorV1;
    // Implements form (1) of uses-allocator construction from the specified
    // 'Alloc' type and exactly 'N' additional arguments. It also provides
    // non-uses allocator construction from 'N' arguments. This test type
    // blows up when form (2) of uses-allocator is even considered.

template <class Alloc, std::size_t N>
class UsesAllocatorV2;
    // Implements form (2) of uses-allocator construction from the specified
    // 'Alloc' type and exactly 'N' additional arguments. It also provides
    // non-uses allocator construction from 'N' arguments.

template <class Alloc, std::size_t N>
class UsesAllocatorV3;
    // Implements both form (1) and (2) of uses-allocator construction from
    // the specified 'Alloc' type and exactly 'N' additional arguments. It also
    // provides non-uses allocator construction from 'N' arguments.

template <class Alloc, std::size_t>
class NotUsesAllocator;
    // Implements both form (1) and (2) of uses-allocator construction from
    // the specified 'Alloc' type and exactly 'N' additional arguments. It also
    // provides non-uses allocator construction from 'N' arguments. However
    // 'NotUsesAllocator' never provides a 'allocator_type' typedef so it is
    // never automatically uses-allocator constructed.


template <class ...ArgTypes, class TestType>
bool checkConstruct(TestType& value, UsesAllocatorType form,
                    typename TestType::CtorAlloc const& alloc)
    // Check that 'value' was constructed using the specified 'form' of
    // construction and with the specified 'ArgTypes...'. Additionally
    // check that 'value' was constructed using the specified 'alloc'.
{
    if (form == UA_None) {
        return value.template checkConstruct<ArgTypes&&...>(form);
    } else {
        return value.template checkConstruct<ArgTypes&&...>(form, alloc);
    }
}


template <class ...ArgTypes, class TestType>
bool checkConstruct(TestType& value, UsesAllocatorType form) {
    return value.template checkConstruct<ArgTypes&&...>(form);
}

template <class TestType>
bool checkConstructionEquiv(TestType& T, TestType& U)
    // check that 'T' and 'U' where initialized in the exact same manner.
{
    return T.checkConstructEquiv(U);
}

////////////////////////////////////////////////////////////////////////////////
namespace detail {

template <bool IsZero, size_t N, class ArgList, class ...Args>
struct TakeNImp;

template <class ArgList, class ...Args>
struct TakeNImp<true, 0, ArgList, Args...> {
  typedef ArgList type;
};

template <size_t N, class ...A1, class F, class ...R>
struct TakeNImp<false, N, ArgumentListID<A1...>, F, R...>
    : TakeNImp<N-1 == 0, N - 1, ArgumentListID<A1..., F>, R...> {};

template <size_t N, class ...Args>
struct TakeNArgs : TakeNImp<N == 0, N, ArgumentListID<>, Args...> {};

template <class T>
struct Identity { typedef T type; };

template <class T>
using IdentityT = typename Identity<T>::type;

template <bool Value>
using EnableIfB = typename std::enable_if<Value, bool>::type;

} // end namespace detail

using detail::EnableIfB;

struct AllocLastTag {};

template <class Alloc>
struct UsesAllocatorTestBase {
public:
    using CtorAlloc = typename std::conditional<
        std::is_same<Alloc, std::experimental::erased_type>::value,
        std::experimental::pmr::memory_resource*,
        Alloc
    >::type;

    template <class ...ArgTypes>
    bool checkConstruct(UsesAllocatorType expectType) const {
        return expectType == constructor_called &&
               makeArgumentID<ArgTypes...>() == *args_id;
    }

    template <class ...ArgTypes>
    bool checkConstruct(UsesAllocatorType expectType,
                        CtorAlloc const& expectAlloc) const {
        return expectType == constructor_called &&
               makeArgumentID<ArgTypes...>() == *args_id &&
               expectAlloc == allocator;
    }

    bool checkConstructEquiv(UsesAllocatorTestBase& O) const {
        return constructor_called == O.constructor_called
            && *args_id == *O.args_id
            && allocator == O.allocator;
    }

protected:
    explicit UsesAllocatorTestBase(const TypeID* aid)
        : args_id(aid), constructor_called(UA_None), allocator()
    {}

    template <class ...Args>
    UsesAllocatorTestBase(std::allocator_arg_t, CtorAlloc const& a, Args&&...)
        : args_id(&makeArgumentID<Args&&...>()),
          constructor_called(UA_AllocArg),
          allocator(a)
    {}

    template <class ...Args>
    UsesAllocatorTestBase(AllocLastTag, Args&&... args)
        : args_id(nullptr),
          constructor_called(UA_AllocLast)
    {
        typedef typename detail::TakeNArgs<sizeof...(Args) - 1, Args&&...>::type
            ArgIDL;
        args_id = &makeTypeID<ArgIDL>();
        getAllocatorFromPack(ArgIDL{}, std::forward<Args>(args)...);
    }

private:
    template <class ...LArgs, class ...Args>
    void getAllocatorFromPack(ArgumentListID<LArgs...>, Args&&... args) {
        getAllocatorFromPackImp<LArgs const&...>(args...);
    }

    template <class ...LArgs>
    void getAllocatorFromPackImp(typename detail::Identity<LArgs>::type..., CtorAlloc const& alloc) {
        allocator = alloc;
    }

    const TypeID* args_id;
    UsesAllocatorType constructor_called = UA_None;
    CtorAlloc allocator;
};

template <class Alloc, size_t Arity>
class UsesAllocatorV1 : public UsesAllocatorTestBase<Alloc>
{
public:
    typedef Alloc allocator_type;

    using Base = UsesAllocatorTestBase<Alloc>;
    using CtorAlloc = typename Base::CtorAlloc;

    UsesAllocatorV1() : Base(&makeArgumentID<>()) {}

    // Non-Uses Allocator Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity> = false>
    UsesAllocatorV1(Args&&... args) : Base(&makeArgumentID<Args&&...>()) {};

    // Uses Allocator Arg Ctor
    template <class ...Args>
    UsesAllocatorV1(std::allocator_arg_t tag, CtorAlloc const & a, Args&&... args)
        : Base(tag, a, std::forward<Args>(args)...)
    { }

    // BLOWS UP: Uses Allocator Last Ctor
    template <class _First, class ...Args, EnableIfB<sizeof...(Args) == Arity> _Dummy = false>
    constexpr UsesAllocatorV1(_First&& __first, Args&&... args)
    {
        static_assert(!std::is_same<_First, _First>::value, "");
    }
};


template <class Alloc, size_t Arity>
class UsesAllocatorV2 : public UsesAllocatorTestBase<Alloc>
{
public:
    typedef Alloc allocator_type;

    using Base = UsesAllocatorTestBase<Alloc>;
    using CtorAlloc = typename Base::CtorAlloc;

    UsesAllocatorV2() : Base(&makeArgumentID<>()) {}

    // Non-Uses Allocator Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity> = false>
    UsesAllocatorV2(Args&&... args) : Base(&makeArgumentID<Args&&...>()) {};

    // Uses Allocator Last Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity + 1> = false>
    UsesAllocatorV2(Args&&... args)
        : Base(AllocLastTag{}, std::forward<Args>(args)...)
    {}
};

template <class Alloc, size_t Arity>
class UsesAllocatorV3 : public UsesAllocatorTestBase<Alloc>
{
public:
    typedef Alloc allocator_type;

    using Base = UsesAllocatorTestBase<Alloc>;
    using CtorAlloc = typename Base::CtorAlloc;

    UsesAllocatorV3() : Base(&makeArgumentID<>()) {}

    // Non-Uses Allocator Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity> = false>
    UsesAllocatorV3(Args&&... args) : Base(&makeArgumentID<Args&&...>()) {};

    // Uses Allocator Arg Ctor
    template <class ...Args>
    UsesAllocatorV3(std::allocator_arg_t tag, CtorAlloc const& alloc, Args&&... args)
        : Base(tag, alloc, std::forward<Args>(args)...)
    {}

    // Uses Allocator Last Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity + 1> = false>
    UsesAllocatorV3(Args&&... args)
        : Base(AllocLastTag{}, std::forward<Args>(args)...)
    {}
};

template <class Alloc, size_t Arity>
class NotUsesAllocator : public UsesAllocatorTestBase<Alloc>
{
public:
    // no allocator_type typedef provided

    using Base = UsesAllocatorTestBase<Alloc>;
    using CtorAlloc = typename Base::CtorAlloc;

    NotUsesAllocator() : Base(&makeArgumentID<>()) {}

    // Non-Uses Allocator Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity> = false>
    NotUsesAllocator(Args&&... args) : Base(&makeArgumentID<Args&&...>()) {};

    // Uses Allocator Arg Ctor
    template <class ...Args>
    NotUsesAllocator(std::allocator_arg_t tag, CtorAlloc const& alloc, Args&&... args)
        : Base(tag, alloc, std::forward<Args>(args)...)
    {}

    // Uses Allocator Last Ctor
    template <class ...Args, EnableIfB<sizeof...(Args) == Arity + 1> = false>
    NotUsesAllocator(Args&&... args)
        : Base(AllocLastTag{}, std::forward<Args>(args)...)
    {}
};

#endif /* USES_ALLOC_TYPES_HPP */
