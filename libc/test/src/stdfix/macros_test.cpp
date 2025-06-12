//===-- Unittests for absfx -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AbsTest.h"
#include "src/stdfix/abshk.h"
#include "src/stdfix/abshr.h"
#include "src/stdfix/absk.h"
#include "src/stdfix/abslk.h"
#include "src/stdfix/abslr.h"
#include "src/stdfix/absr.h"

#include "src/stdfix/countlshk.h"
#include "src/stdfix/countlshr.h"
#include "src/stdfix/countlsk.h"
#include "src/stdfix/countlslk.h"
#include "src/stdfix/countlslr.h"
#include "src/stdfix/countlsr.h"
#include "src/stdfix/countlsuhk.h"
#include "src/stdfix/countlsuhr.h"
#include "src/stdfix/countlsuk.h"
#include "src/stdfix/countlsulk.h"
#include "src/stdfix/countlsulr.h"
#include "src/stdfix/countlsur.h"

#include "src/stdfix/roundhk.h"
#include "src/stdfix/roundhr.h"
#include "src/stdfix/roundk.h"
#include "src/stdfix/roundlk.h"
#include "src/stdfix/roundlr.h"
#include "src/stdfix/roundr.h"
#include "src/stdfix/rounduhk.h"
#include "src/stdfix/rounduhr.h"
#include "src/stdfix/rounduk.h"
#include "src/stdfix/roundulk.h"
#include "src/stdfix/roundulr.h"
#include "src/stdfix/roundur.h"

using LIBC_NAMESPACE::abshk;
using LIBC_NAMESPACE::abshr;
using LIBC_NAMESPACE::absk;
using LIBC_NAMESPACE::abslk;
using LIBC_NAMESPACE::abslr;
using LIBC_NAMESPACE::absr;
using LIBC_NAMESPACE::countlshk;
using LIBC_NAMESPACE::countlshr;
using LIBC_NAMESPACE::countlsk;
using LIBC_NAMESPACE::countlslk;
using LIBC_NAMESPACE::countlslr;
using LIBC_NAMESPACE::countlsr;
using LIBC_NAMESPACE::countlsuhk;
using LIBC_NAMESPACE::countlsuhr;
using LIBC_NAMESPACE::countlsuk;
using LIBC_NAMESPACE::countlsulk;
using LIBC_NAMESPACE::countlsulr;
using LIBC_NAMESPACE::countlsur;
using LIBC_NAMESPACE::roundhk;
using LIBC_NAMESPACE::roundhr;
using LIBC_NAMESPACE::roundk;
using LIBC_NAMESPACE::roundlk;
using LIBC_NAMESPACE::roundlr;
using LIBC_NAMESPACE::roundr;
using LIBC_NAMESPACE::rounduhk;
using LIBC_NAMESPACE::rounduhr;
using LIBC_NAMESPACE::rounduk;
using LIBC_NAMESPACE::roundulk;
using LIBC_NAMESPACE::roundulr;
using LIBC_NAMESPACE::roundur;

TEST(LlvmLibcAbsfxTest, Basic) {
  ASSERT_EQ(absfx(-0.5r), absr(-0.5r));
  ASSERT_EQ(absfx(-0.5hr), abshr(-0.5hr));
  ASSERT_EQ(absfx(-0.5lr), abslr(-0.5lr));
  ASSERT_EQ(absfx(-0.5k), absk(-0.5k));
  ASSERT_EQ(absfx(-0.5hk), abshk(-0.5hk));
  ASSERT_EQ(absfx(-0.5lk), abslk(-0.5lk));
}

TEST(LlvmLibcRoundfxTest, Basic) {
  ASSERT_EQ(roundfx(0.75r, 0), roundr(0.75r, 0));
  ASSERT_EQ(roundfx(0.75hr, 0), roundhr(0.75hr, 0));
  ASSERT_EQ(roundfx(0.75lr, 0), roundlr(0.75lr, 0));
  ASSERT_EQ(roundfx(0.75k, 0), roundk(0.75k, 0));
  ASSERT_EQ(roundfx(0.75hk, 0), roundhk(0.75hk, 0));
  ASSERT_EQ(roundfx(0.75lk, 0), roundlk(0.75lk, 0));

  ASSERT_EQ(roundfx(0.75ur, 0), roundur(0.75ur, 0));
  ASSERT_EQ(roundfx(0.75uhr, 0), rounduhr(0.75uhr, 0));
  ASSERT_EQ(roundfx(0.75ulr, 0), roundulr(0.75ulr, 0));
  ASSERT_EQ(roundfx(0.75uk, 0), rounduk(0.75uk, 0));
  ASSERT_EQ(roundfx(0.75uhk, 0), rounduhk(0.75uhk, 0));
  ASSERT_EQ(roundfx(0.75ulk, 0), roundulk(0.75ulk, 0));
}

TEST(LlvmLibcCountlsfxTest, Basic) {
  ASSERT_EQ(countlsfx(0.5r), countlsr(0.5r));
  ASSERT_EQ(countlsfx(0.5hr), countlshr(0.5hr));
  ASSERT_EQ(countlsfx(0.5lr), countlslr(0.5lr));
  ASSERT_EQ(countlsfx(0.5k), countlsk(0.5k));
  ASSERT_EQ(countlsfx(0.5hk), countlshk(0.5hk));
  ASSERT_EQ(countlsfx(0.5lk), countlslk(0.5lk));

  ASSERT_EQ(countlsfx(0.5ur), countlsur(0.5ur));
  ASSERT_EQ(countlsfx(0.5uhr), countlsuhr(0.5uhr));
  ASSERT_EQ(countlsfx(0.5ulr), countlsulr(0.5ulr));
  ASSERT_EQ(countlsfx(0.5uk), countlsuk(0.5uk));
  ASSERT_EQ(countlsfx(0.5uhk), countlsuhk(0.5uhk));
  ASSERT_EQ(countlsfx(0.5ulk), countlsulk(0.5ulk));
}
