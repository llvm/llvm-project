//===- WMMALoweringTest.cpp - matrix fragment mapping tests --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/wmma-lowering.h"

#include "gtest/gtest.h"

#include <tuple>

using namespace COMGR::transpiler;

namespace {

unsigned identity(unsigned Lane, unsigned Dword, unsigned Element) {
  return (Lane * 8 + Dword) * 4 + Element;
}

std::tuple<unsigned, unsigned, unsigned> decodeIdentity(unsigned Identity) {
  unsigned Element = Identity % 4;
  Identity /= 4;
  return {Identity / 8, Identity % 8, Element};
}

unsigned getWMMAInputK(unsigned Lane, unsigned Dword, unsigned Element,
                       unsigned ElementBits) {
  unsigned LaneHalf = (Lane % 32) / 16;
  if (ElementBits == 16)
    return (Dword / 4) * 16 + LaneHalf * 8 + (Dword % 4) * 2 + Element;
  return (Dword / 4) * 32 + ((Dword / 2) % 2) * 16 + LaneHalf * 8 +
         (Dword % 2) * 4 + Element;
}

void checkInputMapping(unsigned ElementBits) {
  unsigned ElementsPerDword = 32 / ElementBits;
  unsigned ElementsPerMFMA = ElementBits == 16 ? 16 : 32;

  for (unsigned SourceWave = 0; SourceWave != 2; ++SourceWave) {
    for (unsigned KHalf = 0; KHalf != 2; ++KHalf) {
      for (unsigned TargetLane = 0; TargetLane != 64; ++TargetLane) {
        unsigned LaneGroup = TargetLane / 16;
        for (unsigned TargetDword = 0; TargetDword != 2; ++TargetDword) {
          detail::FragmentDwordLocation Source = detail::getWMMAInputDword(
              SourceWave, TargetLane, TargetDword, KHalf, 2);
          for (unsigned Element = 0; Element != ElementsPerDword; ++Element) {
            auto [Lane, Dword, Subelement] =
                decodeIdentity(identity(Source.Lane, Source.Dword, Element));
            EXPECT_EQ(Lane / 32, SourceWave);
            EXPECT_EQ(Lane % 16, TargetLane % 16);

            unsigned TargetK = LaneGroup * 2 * ElementsPerDword +
                               TargetDword * ElementsPerDword + Subelement;
            unsigned ExpectedK = TargetK;
            if (ElementBits == 8) {
              unsigned Quarter = LaneGroup;
              ExpectedK = ((Quarter & 1) * 2 + Quarter / 2) * 8 +
                          TargetDword * 4 + Subelement;
            }
            ExpectedK += KHalf * ElementsPerMFMA;
            EXPECT_EQ(getWMMAInputK(Lane, Dword, Subelement, ElementBits),
                      ExpectedK);
          }
        }
      }
    }
  }
}

} // namespace

TEST(WMMALoweringTest, RedistributesF32Inputs) {
  for (unsigned SourceWave = 0; SourceWave != 2; ++SourceWave) {
    for (unsigned K = 0; K != 4; ++K) {
      for (unsigned Row = 0; Row != 16; ++Row) {
        detail::FragmentDwordLocation Source =
            detail::getWMMAInputDword(SourceWave, Row + 16 * K, 0, 0, 1);
        EXPECT_EQ(Source.Lane, 32 * SourceWave + Row + 16 * (K / 2));
        EXPECT_EQ(Source.Dword, K % 2);
      }
    }
  }
}

TEST(WMMALoweringTest, RedistributesF16Inputs) { checkInputMapping(16); }

TEST(WMMALoweringTest, RedistributesI8Inputs) { checkInputMapping(8); }

TEST(WMMALoweringTest, RedistributesAccumulatorRows) {
  for (unsigned SourceWave = 0; SourceWave != 2; ++SourceWave) {
    for (unsigned TargetLane = 0; TargetLane != 64; ++TargetLane) {
      for (unsigned TargetDword = 0; TargetDword != 4; ++TargetDword) {
        detail::FragmentDwordLocation Source = detail::getWMMAAccumulatorDword(
            SourceWave, TargetLane, TargetDword);
        auto [Lane, Dword, Element] =
            decodeIdentity(identity(Source.Lane, Source.Dword, 0));
        EXPECT_EQ(Element, 0u);
        EXPECT_EQ(Lane / 32, SourceWave);
        EXPECT_EQ(Lane % 16, TargetLane % 16);
        EXPECT_EQ(Dword + 8 * ((Lane % 32) / 16),
                  4 * (TargetLane / 16) + TargetDword);
      }
    }
  }
}

TEST(WMMALoweringTest, CollectsResultDwords) {
  for (unsigned SourceLane = 0; SourceLane != 32; ++SourceLane) {
    for (unsigned SourceDword = 0; SourceDword != 8; ++SourceDword) {
      detail::FragmentDwordLocation Source =
          detail::getMFMAResultDword(SourceLane, SourceDword);
      auto [Lane, Dword, Element] =
          decodeIdentity(identity(Source.Lane, Source.Dword, 0));
      EXPECT_EQ(Element, 0u);
      EXPECT_EQ(Lane % 16, SourceLane % 16);
      EXPECT_EQ(4 * (Lane / 16) + Dword, SourceDword + 8 * (SourceLane / 16));
    }
  }
}
