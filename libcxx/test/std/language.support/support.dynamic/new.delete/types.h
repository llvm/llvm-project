//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_LANGUAGE_SUPPORT_SUPPORT_DYNAMIC_NEW_DELETE_TYPES_H
#define TEST_STD_LANGUAGE_SUPPORT_SUPPORT_DYNAMIC_NEW_DELETE_TYPES_H

#include <cstddef>

#include "test_macros.h"

struct LifetimeInformation {
    void* address_constructed = nullptr;
    void* address_destroyed = nullptr;
};

struct TrackLifetime {
    TrackLifetime(LifetimeInformation& info) : info_(&info) {
        info_->address_constructed = this;
    }
    ~TrackLifetime() {
        info_->address_destroyed = this;
    }
    LifetimeInformation* info_;
};

#if TEST_STD_VER >= 17
struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) TrackLifetimeOverAligned {
    TrackLifetimeOverAligned(LifetimeInformation& info) : info_(&info) {
        info_->address_constructed = this;
    }
    ~TrackLifetimeOverAligned() {
        info_->address_destroyed = this;
    }
    LifetimeInformation* info_;
};

struct alignas(std::max_align_t) TrackLifetimeMaxAligned {
    TrackLifetimeMaxAligned(LifetimeInformation& info) : info_(&info) {
        info_->address_constructed = this;
    }
    ~TrackLifetimeMaxAligned() {
        info_->address_destroyed = this;
    }
    LifetimeInformation* info_;
};

struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) OverAligned {

};

struct alignas(std::max_align_t) MaxAligned {

};
#endif // TEST_STD_VER >= 17

#endif // TEST_STD_LANGUAGE_SUPPORT_SUPPORT_DYNAMIC_NEW_DELETE_TYPES_H
