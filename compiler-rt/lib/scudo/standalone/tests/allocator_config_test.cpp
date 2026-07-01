//===-- allocator_config_test.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "allocator_config.h"
#include "allocator_config_wrapper.h"
#include "common.h"
#include "secondary.h"
#include "string_utils.h"

#include <type_traits>

struct TestBaseConfig {
  template <typename> using TSDRegistryT = void;
  template <typename> using PrimaryT = void;
  template <typename> using SecondaryT = void;
};

struct TestBaseConfigEnableOptionalFlag : public TestBaseConfig {
  static const bool MaySupportMemoryTagging = true;
  // Use the getter to avoid the test to `use` the address of static const
  // variable (which requires additional explicit definition).
  static bool getMaySupportMemoryTagging() { return MaySupportMemoryTagging; }
};

struct TestBasePrimaryConfig {
  using SizeClassMap = void;
  static const scudo::uptr RegionSizeLog = 18U;
  static const scudo::uptr GroupSizeLog = 18U;
  static const scudo::s32 MinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 MaxReleaseToOsIntervalMs = INT32_MAX;
  typedef scudo::uptr CompactPtrT;
  static const scudo::uptr CompactPtrScale = 0;
  static const scudo::uptr MapSizeIncrement = 1UL << 18;
};

struct TestPrimaryConfig : public TestBaseConfig {
  struct Primary : TestBasePrimaryConfig {};
};

struct TestPrimaryConfigEnableOptionalFlag : public TestBaseConfig {
  struct Primary : TestBasePrimaryConfig {
    static const bool EnableRandomOffset = true;
    static bool getEnableRandomOffset() { return EnableRandomOffset; }
  };
};

struct TestPrimaryConfigEnableOptionalType : public TestBaseConfig {
  struct DummyConditionVariable {};

  struct Primary : TestBasePrimaryConfig {
    using ConditionVariableT = DummyConditionVariable;
  };
};

struct TestSecondaryConfig : public TestPrimaryConfig {
  struct Secondary {
    template <typename Config>
    using CacheT = scudo::MapAllocatorNoCache<Config>;
  };
};

struct TestSecondaryCacheConfigEnableOptionalFlag : public TestPrimaryConfig {
  struct Secondary {
    struct Cache {
      static const scudo::u32 EntriesArraySize = 256U;
      static scudo::u32 getEntriesArraySize() { return EntriesArraySize; }
    };
    template <typename T> using CacheT = scudo::MapAllocatorCache<T>;
  };
};

TEST(ScudoAllocatorConfigTest, VerifyOptionalFlags) {
  // Test the top level allocator optional config.
  //
  // `MaySupportMemoryTagging` is default off.
  EXPECT_FALSE(scudo::BaseConfig<TestBaseConfig>::getMaySupportMemoryTagging());
  EXPECT_EQ(scudo::BaseConfig<
                TestBaseConfigEnableOptionalFlag>::getMaySupportMemoryTagging(),
            TestBaseConfigEnableOptionalFlag::getMaySupportMemoryTagging());

  EXPECT_FALSE(scudo::BaseConfig<TestBaseConfig>::getQuarantineDisabled());
  EXPECT_TRUE(scudo::BaseConfig<TestBaseConfig>::getExactUsableSize());
  EXPECT_TRUE(
      scudo::BaseConfig<TestBaseConfig>::getAbortOnDeallocTypeMismatch());
  EXPECT_TRUE(
      scudo::BaseConfig<TestBaseConfig>::getAbortOnDeallocSizeMismatch());
  EXPECT_TRUE(
      scudo::BaseConfig<TestBaseConfig>::getAbortOnDeallocAlignmentMismatch());

  // Test primary optional config.
  //
  // `EnableRandomeOffset` is default off.
  EXPECT_FALSE(
      scudo::PrimaryConfig<TestPrimaryConfig>::getEnableRandomOffset());
  EXPECT_EQ(
      scudo::PrimaryConfig<
          TestPrimaryConfigEnableOptionalFlag>::getEnableRandomOffset(),
      TestPrimaryConfigEnableOptionalFlag::Primary::getEnableRandomOffset());

  // `ConditionVariableT` is default off.
  EXPECT_FALSE(
      scudo::PrimaryConfig<TestPrimaryConfig>::hasConditionVariableT());
  EXPECT_TRUE(scudo::PrimaryConfig<
              TestPrimaryConfigEnableOptionalType>::hasConditionVariableT());
  EXPECT_TRUE((std::is_same_v<
               typename scudo::PrimaryConfig<
                   TestPrimaryConfigEnableOptionalType>::ConditionVariableT,
               typename TestPrimaryConfigEnableOptionalType::Primary::
                   ConditionVariableT>));

  EXPECT_TRUE(scudo::PrimaryConfig<TestPrimaryConfig>::getEnableBlockCache());
  EXPECT_EQ(scudo::PrimaryConfig<TestPrimaryConfig>::getCompactPtrScale(), 0U);
  EXPECT_TRUE(
      scudo::PrimaryConfig<TestPrimaryConfig>::getEnableContiguousRegions());

  // Test secondary cache optional config.
  using NoCacheConfig =
      scudo::SecondaryConfig<TestSecondaryConfig>::CacheConfig;
  // `EntriesArraySize` is default 0.
  EXPECT_EQ(NoCacheConfig::getEntriesArraySize(), 0U);

  using CacheConfig = scudo::SecondaryConfig<
      TestSecondaryCacheConfigEnableOptionalFlag>::CacheConfig;
  EXPECT_EQ(CacheConfig::getEntriesArraySize(),
            TestSecondaryCacheConfigEnableOptionalFlag::Secondary::Cache::
                getEntriesArraySize());

  EXPECT_TRUE(
      scudo::SecondaryConfig<TestSecondaryConfig>::getEnableGuardPages());

  EXPECT_EQ(CacheConfig::getQuarantineSize(), 0U);
  EXPECT_EQ(CacheConfig::getDefaultMaxEntriesCount(), 0U);
  EXPECT_EQ(CacheConfig::getDefaultMaxEntrySize(), 0U);
  EXPECT_EQ(CacheConfig::getMinReleaseToOsIntervalMs(), INT32_MIN);
  EXPECT_EQ(CacheConfig::getMaxReleaseToOsIntervalMs(), INT32_MAX);
  EXPECT_EQ(CacheConfig::getDefaultReleaseToOsIntervalMs(), INT32_MIN);
}
