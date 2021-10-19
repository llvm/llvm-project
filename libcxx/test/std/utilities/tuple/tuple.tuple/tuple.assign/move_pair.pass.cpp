//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class U1, class U2>
//   tuple& operator=(pair<U1, U2>&& u);

// UNSUPPORTED: c++03

#include <tuple>
#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"

struct B
{
    int id_;

    explicit B(int i = 0) : id_(i) {}

    virtual ~B() {}
};

struct D
    : B
{
    explicit D(int i) : B(i) {}
};

struct TrackMove
{
    TrackMove() : value(0), moved_from(false) { }
    explicit TrackMove(int v) : value(v), moved_from(false) { }
    TrackMove(TrackMove const& other) : value(other.value), moved_from(false) { }
    TrackMove(TrackMove&& other) : value(other.value), moved_from(false) {
        other.moved_from = true;
    }
    TrackMove& operator=(TrackMove const& other) {
        value = other.value;
        moved_from = false;
        return *this;
    }
    TrackMove& operator=(TrackMove&& other) {
        value = other.value;
        moved_from = false;
        other.moved_from = true;
        return *this;
    }

    int value;
    bool moved_from;
};

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};

struct MoveAssignable
{
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CopyAssignable
{
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};

struct NothrowMoveAssignable
{
    NothrowMoveAssignable& operator=(NothrowMoveAssignable&&) noexcept { return *this; }
};

struct PotentiallyThrowingMoveAssignable
{
    PotentiallyThrowingMoveAssignable& operator=(PotentiallyThrowingMoveAssignable&&) { return *this; }
};

int main(int, char**)
{
    {
        typedef std::pair<long, std::unique_ptr<D>> T0;
        typedef std::tuple<long long, std::unique_ptr<B>> T1;
        T0 t0(2, std::unique_ptr<D>(new D(3)));
        T1 t1;
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1)->id_ == 3);
    }
    {
        using T = std::tuple<int, NonAssignable>;
        using P = std::pair<int, NonAssignable>;
        static_assert(!std::is_assignable<T&, P&&>::value, "");
    }
    {
      using T = std::tuple<int, int, int>;
      using P = std::pair<int, int>;
      static_assert(!std::is_assignable<T&, P&&>::value, "");
    }
    {
        typedef std::tuple<NothrowMoveAssignable, long> Tuple;
        typedef std::pair<NothrowMoveAssignable, int> Pair;
        static_assert(std::is_nothrow_assignable<Tuple&, Pair&&>::value, "");
        static_assert(!std::is_assignable<Tuple&, Pair const&&>::value, "");
    }
    {
        typedef std::tuple<PotentiallyThrowingMoveAssignable, long> Tuple;
        typedef std::pair<PotentiallyThrowingMoveAssignable, int> Pair;
        static_assert(std::is_assignable<Tuple&, Pair&&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair&&>::value, "");
        static_assert(!std::is_assignable<Tuple&, Pair const&&>::value, "");
    }
    {
        // We assign through the reference and don't move out of the incoming ref,
        // so this doesn't work (but would if the type were CopyAssignable).
        {
            using T = std::tuple<MoveAssignable&, int>;
            using P = std::pair<MoveAssignable&, int>;
            static_assert(!std::is_assignable<T&, P&&>::value, "");
        }

        // ... works if it's CopyAssignable
        {
            using T = std::tuple<CopyAssignable&, int>;
            using P = std::pair<CopyAssignable&, int>;
            static_assert(std::is_assignable<T&, P&&>::value, "");
        }

        // For rvalue-references, we can move-assign if the type is MoveAssignable
        // or CopyAssignable (since in the worst case the move will decay into a copy).
        {
            using T1 = std::tuple<MoveAssignable&&, int>;
            using P1 = std::pair<MoveAssignable&&, int>;
            static_assert(std::is_assignable<T1&, P1&&>::value, "");

            using T2 = std::tuple<CopyAssignable&&, int>;
            using P2 = std::pair<CopyAssignable&&, int>;
            static_assert(std::is_assignable<T2&, P2&&>::value, "");
        }

        // In all cases, we can't move-assign if the types are not assignable,
        // since we assign through the reference.
        {
            using T1 = std::tuple<NonAssignable&, int>;
            using P1 = std::pair<NonAssignable&, int>;
            static_assert(!std::is_assignable<T1&, P1&&>::value, "");

            using T2 = std::tuple<NonAssignable&&, int>;
            using P2 = std::pair<NonAssignable&&, int>;
            static_assert(!std::is_assignable<T2&, P2&&>::value, "");
        }
    }
    {
        // Make sure that we don't incorrectly move out of the source's reference.
        using Dest = std::tuple<TrackMove, int>;
        using Source = std::pair<TrackMove&, int>;
        TrackMove track{3};
        Source src(track, 4);
        assert(!track.moved_from);

        Dest dst;
        dst = std::move(src); // here we should make a copy
        assert(!track.moved_from);
        assert(std::get<0>(dst).value == 3);
    }
    {
        // But we do move out of the source's reference if it's a rvalue ref
        using Dest = std::tuple<TrackMove, int>;
        using Source = std::pair<TrackMove&&, int>;
        TrackMove track{3};
        Source src(std::move(track), 4);
        assert(!track.moved_from); // we just took a reference

        Dest dst;
        dst = std::move(src);
        assert(track.moved_from);
        assert(std::get<0>(dst).value == 3);
    }
    {
        // If the pair holds a value, then we move out of it too
        using Dest = std::tuple<TrackMove, int>;
        using Source = std::pair<TrackMove, int>;
        Source src(TrackMove{3}, 4);
        Dest dst;
        dst = std::move(src);
        assert(src.first.moved_from);
        assert(std::get<0>(dst).value == 3);
    }

    return 0;
}
