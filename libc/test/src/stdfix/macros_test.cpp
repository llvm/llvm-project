//===-- Unittests for absfx -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AbsTest.h"
#include "src/stdfix/absr.h"
#include "src/stdfix/abshr.h"
#include "src/stdfix/abslr.h"
#include "src/stdfix/absk.h"
#include "src/stdfix/abshk.h"
#include "src/stdfix/abslk.h"

#include "src/stdfix/countlsr.h"
#include "src/stdfix/countlshr.h"
#include "src/stdfix/countlslr.h"
#include "src/stdfix/countlsk.h"
#include "src/stdfix/countlshk.h"
#include "src/stdfix/countlslk.h"
#include "src/stdfix/countlsur.h"
#include "src/stdfix/countlsuhr.h"
#include "src/stdfix/countlsulr.h"
#include "src/stdfix/countlsuk.h"
#include "src/stdfix/countlsuhk.h"
#include "src/stdfix/countlsulk.h"

#include "src/stdfix/roundr.h"
#include "src/stdfix/roundhr.h"
#include "src/stdfix/roundlr.h"
#include "src/stdfix/roundk.h"
#include "src/stdfix/roundhk.h"
#include "src/stdfix/roundlk.h"
#include "src/stdfix/roundur.h"
#include "src/stdfix/rounduhr.h"
#include "src/stdfix/roundulr.h"
#include "src/stdfix/rounduk.h"
#include "src/stdfix/rounduhk.h"
#include "src/stdfix/roundulk.h"

TEST(LlvmLibcAbsfxTest, Basic) {
  ASSERT_EQ(absfx(-0.5r), LIBC_NAMESPACE::absr(-0.5r));
  ASSERT_EQ(absfx(-0.5hr), LIBC_NAMESPACE::abshr(-0.5hr));
  ASSERT_EQ(absfx(-0.5lr), LIBC_NAMESPACE::abslr(-0.5lr));
  ASSERT_EQ(absfx(-0.5k), LIBC_NAMESPACE::absk(-0.5k));
  ASSERT_EQ(absfx(-0.5hk), LIBC_NAMESPACE::abshk(-0.5hk));
  ASSERT_EQ(absfx(-0.5lk), LIBC_NAMESPACE::abslk(-0.5lk));
}

TEST(LlvmLibcRoundfxTest, Basic) {
  ASSERT_EQ(roundfx(0.75r, 0), LIBC_NAMESPACE::roundr(0.75r, 0));
  ASSERT_EQ(roundfx(0.75hr, 0), LIBC_NAMESPACE::roundhr(0.75hr, 0));
  ASSERT_EQ(roundfx(0.75lr, 0), LIBC_NAMESPACE::roundlr(0.75lr, 0));
  ASSERT_EQ(roundfx(0.75k, 0), LIBC_NAMESPACE::roundk(0.75k, 0));
  ASSERT_EQ(roundfx(0.75hk, 0), LIBC_NAMESPACE::roundhk(0.75hk, 0));
  ASSERT_EQ(roundfx(0.75lk, 0), LIBC_NAMESPACE::roundlk(0.75lk, 0));

  ASSERT_EQ(roundfx(0.75ur, 0), LIBC_NAMESPACE::roundur(0.75ur, 0));
  ASSERT_EQ(roundfx(0.75uhr, 0), LIBC_NAMESPACE::rounduhr(0.75uhr, 0));
  ASSERT_EQ(roundfx(0.75ulr, 0), LIBC_NAMESPACE::roundulr(0.75ulr, 0));
  ASSERT_EQ(roundfx(0.75uk, 0), LIBC_NAMESPACE::rounduk(0.75uk, 0));
  ASSERT_EQ(roundfx(0.75uhk, 0), LIBC_NAMESPACE::rounduhk(0.75uhk, 0));
  ASSERT_EQ(roundfx(0.75ulk, 0), LIBC_NAMESPACE::roundulk(0.75ulk, 0));
}


TEST(LlvmLibcCountlsfxTest, Basic) {
  ASSERT_EQ(countlsfx(0.5r), LIBC_NAMESPACE::countlsr(0.5r));
  ASSERT_EQ(countlsfx(0.5hr), LIBC_NAMESPACE::countlshr(0.5hr));
  ASSERT_EQ(countlsfx(0.5lr), LIBC_NAMESPACE::countlslr(0.5lr));
  ASSERT_EQ(countlsfx(0.5k), LIBC_NAMESPACE::countlsk(0.5k));
  ASSERT_EQ(countlsfx(0.5hk), LIBC_NAMESPACE::countlshk(0.5hk));
  ASSERT_EQ(countlsfx(0.5lk), LIBC_NAMESPACE::countlslk(0.5lk));

  ASSERT_EQ(countlsfx(0.5ur), LIBC_NAMESPACE::countlsr(0.5ur));
  ASSERT_EQ(countlsfx(0.5uhr), LIBC_NAMESPACE::countlshr(0.5uhr));
  ASSERT_EQ(countlsfx(0.5ulr), LIBC_NAMESPACE::countlslr(0.5ulr));
  ASSERT_EQ(countlsfx(0.5uk), LIBC_NAMESPACE::countlsk(0.5uk));
  ASSERT_EQ(countlsfx(0.5uhk), LIBC_NAMESPACE::countlshk(0.5uhk));
  ASSERT_EQ(countlsfx(0.5ulk), LIBC_NAMESPACE::countlslk(0.5ulk));
}

