//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_TRACKED_VALUE_H
#define SUPPORT_TRACKED_VALUE_H

#include <cassert>

struct TrackedValue {
    enum State { CONSTRUCTED, MOVED_FROM, DESTROYED };
    State state;

    TrackedValue() : state(State::CONSTRUCTED) {}

    TrackedValue(TrackedValue const& t) : state(State::CONSTRUCTED) {
        assert(t.state != State::MOVED_FROM && "copying a moved-from object");
        assert(t.state != State::DESTROYED  && "copying a destroyed object");
    }

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    TrackedValue(TrackedValue&& t) : state(State::CONSTRUCTED) {
        assert(t.state != State::MOVED_FROM && "double moving from an object");
        assert(t.state != State::DESTROYED  && "moving from a destroyed object");
        t.state = State::MOVED_FROM;
    }
#endif

    TrackedValue& operator=(TrackedValue const& t) {
        assert(state != State::DESTROYED && "copy assigning into destroyed object");
        assert(t.state != State::MOVED_FROM && "copying a moved-from object");
        assert(t.state != State::DESTROYED  && "copying a destroyed object");
        state = t.state;
        return *this;
    }

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    TrackedValue& operator=(TrackedValue&& t) {
        assert(state != State::DESTROYED && "move assigning into destroyed object");
        assert(t.state != State::MOVED_FROM && "double moving from an object");
        assert(t.state != State::DESTROYED  && "moving from a destroyed object");
        state = t.state;
        t.state = State::MOVED_FROM;
        return *this;
    }
#endif

    ~TrackedValue() {
        assert(state != State::DESTROYED && "double-destroying an object");
        state = State::DESTROYED;
    }
};

#endif // SUPPORT_TRACKED_VALUE_H
