/*
 * Copyright (C) 2018 Cavium, Inc.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#if defined(TARGET_ARM64)
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <arm_neon.h>

/*
 * https://www.ibm.com/support/knowledgecenter/SSGH2K_13.1.2/com.ibm.xlc131.aix.doc/compiler_ref/vec_intrin_cpp.html
 */

struct __s128f {
  typedef float  vrs4_t __attribute__((vector_size(4 * sizeof(float))));
  typedef double vrd2_t __attribute__((vector_size(2 * sizeof(double))));
  union {
    float         xf[4];
    int           xi[4];
    unsigned int  xui[4];
    double        xd[2];
    vrs4_t        xfrd;
    vrd2_t        xdrd;
    float32x4_t   nfrd;
    float64x2_t   ndrd;
    long double   b;
    unsigned char c[16];
  } __attribute__((aligned(16)));

  __s128f() : b(0) { }

  __s128f(const float f[4]) : b(0) {
    xfrd[0] = f[0];
    xfrd[1] = f[1];
    xfrd[2] = f[2];
    xfrd[3] = f[3];
  }

  __s128f(const double d[2]) : b(0) {
    xdrd[0] = d[0];
    xdrd[1] = d[1];
  }

  __s128f(const int i[4]) : b(0) {
    xfrd[0] = i[0];
    xfrd[1] = i[1];
    xfrd[2] = i[2];
    xfrd[3] = i[3];
  }

  __s128f(const unsigned int i[4]) : b(0) {
    xfrd[0] = i[0];
    xfrd[1] = i[1];
    xfrd[2] = i[2];
    xfrd[3] = i[3];
  }

  __s128f(float e0, float e1, float e2, float e3) : b(0) {
    xfrd[0] = e0;
    xfrd[1] = e1;
    xfrd[2] = e2;
    xfrd[3] = e3;
  }

  __s128f(const long double& v) : b(v) { }

  __s128f(const vrs4_t& v) : xfrd(v) { }

  __s128f(const vrd2_t& v) : xdrd(v) { }

  __s128f(const __s128f& rhs) : b(rhs.b) { }

  inline __s128f& operator=(const __s128f& rhs) {
    if (this != &rhs)
      xfrd = rhs.xfrd;

    return *this;
  }

  inline __s128f& operator=(long double ld) {
    b = ld;
    return *this;
  }

  inline __s128f& operator=(const vrs4_t& rhs) {
    xfrd = rhs;
    return *this;
  }

  inline __s128f& operator=(const vrd2_t& rhs) {
    xdrd = rhs;
    return *this;
  }

  inline operator long double() const {
    return b;
  }

  inline operator vrs4_t() const {
    return xfrd;
  }

  inline operator vrd2_t() const {
    return xdrd;
  }

  inline operator bool() const {
    return xfrd[0] == 0.0f && xfrd[1] == 0.0f &&
      xfrd[2] == 0.0f && xfrd[3] == 0.0f;
  }

  inline __s128f operator+(float f) const {
    __s128f r(*this);

    if (f != 0.0) {
      r.xfrd[0] += f;
      r.xfrd[1] += f;
      r.xfrd[2] += f;
      r.xfrd[3] += f;
    }

    return r;
  }

  inline __s128f operator+(double d) const {
    __s128f r(*this);

    if (d != 0.0) {
      r.xdrd[0] += d;
      r.xdrd[1] += d;
    }

    return r;
  }

  inline __s128f operator+(unsigned int i) const {
    __s128f r(*this);

    if (i != 0U) {
      r.xfrd[0] += i;
      r.xfrd[1] += i;
      r.xfrd[2] += i;
      r.xfrd[3] += i;
    }

    return r;
  }

  inline __s128f operator+(int i) const {
    __s128f r(*this);

    if (i != 0) {
      r.xfrd[0] += i;
      r.xfrd[1] += i;
      r.xfrd[2] += i;
      r.xfrd[3] += i;
    }

    return r;
  }

  inline __s128f operator+(const long double& ld) const {
    __s128f r(*this);

    if (ld != 0.0)
      r.b += ld;

    return r;
  }

  inline __s128f operator+(const __s128f& s) const {
    __s128f r(*this);

    r.xfrd[0] += s.xfrd[0];
    r.xfrd[1] += s.xfrd[1];
    r.xfrd[2] += s.xfrd[2];
    r.xfrd[3] += s.xfrd[3];

    return r;
  }

  inline __s128f operator-(const __s128f& s) const {
    __s128f r(*this);

    r.xfrd[0] -= s.xfrd[0];
    r.xfrd[1] -= s.xfrd[1];
    r.xfrd[2] -= s.xfrd[2];
    r.xfrd[3] -= s.xfrd[3];

    return r;
  }

  inline __s128f operator-() const {
    return __s128f(-xfrd[0], -xfrd[1], -xfrd[2], -xfrd[3]);
  }

  inline __s128f operator*(const __s128f& rhs) const {
    __s128f r(*this);

    r.xfrd[0] *= rhs.xfrd[0];
    r.xfrd[1] *= rhs.xfrd[1];
    r.xfrd[2] *= rhs.xfrd[2];
    r.xfrd[3] *= rhs.xfrd[3];

    return r;
  }

  inline __s128f operator/(const __s128f& rhs) const {
    __s128f r(*this);

    r.xfrd[0] /= rhs.xfrd[0];
    r.xfrd[1] /= rhs.xfrd[1];
    r.xfrd[2] /= rhs.xfrd[2];
    r.xfrd[3] /= rhs.xfrd[3];

    return r;
  }

  inline __s128f& operator+=(unsigned int i) {
    if (i != 0U) {
      xfrd[0] += i;
      xfrd[1] += i;
      xfrd[2] += i;
      xfrd[3] += i;
    }

    return *this;
  }

  inline __s128f& operator+=(int i) {
    if (i != 0) {
      xfrd[0] += i;
      xfrd[1] += i;
      xfrd[2] += i;
      xfrd[3] += i;
    }

    return *this;
  }

  inline __s128f& operator+=(float f) {
    if (f != 0.0) {
      xfrd[0] += f;
      xfrd[1] += f;
      xfrd[2] += f;
      xfrd[3] += f;
    }

    return *this;
  }

  inline __s128f& operator+=(double d) {
    if (d != 0.0) {
      xfrd[0] += (float) d;
      xfrd[1] += (float) d;
      xfrd[2] += (float) d;
      xfrd[3] += (float) d;
    }

    return *this;
  }

  inline __s128f& operator+=(long double ld) {
    if (ld != 0.0)
      b += ld;

    return *this;
  }

  inline __s128f& operator+=(const __s128f& rhs) {
    xfrd[0] += rhs.xfrd[0];
    xfrd[1] += rhs.xfrd[1];
    xfrd[2] += rhs.xfrd[2];
    xfrd[3] += rhs.xfrd[3];

    return *this;
  }

  inline __s128f& operator/=(const __s128f& rhs) {
    xfrd[0] /= rhs.xfrd[0];
    xfrd[1] /= rhs.xfrd[1];
    xfrd[2] /= rhs.xfrd[2];
    xfrd[3] /= rhs.xfrd[3];

    return *this;
  }

  inline __s128f& operator*=(const __s128f& rhs) {
    xfrd[0] *= rhs.xfrd[0];
    xfrd[1] *= rhs.xfrd[1];
    xfrd[2] *= rhs.xfrd[2];
    xfrd[3] *= rhs.xfrd[3];

    return *this;
  }

  inline __s128f& operator-=(const __s128f& rhs) {
    xfrd[0] -= rhs.xfrd[0];
    xfrd[1] -= rhs.xfrd[1];
    xfrd[2] -= rhs.xfrd[2];
    xfrd[3] -= rhs.xfrd[3];

    return *this;
  }
} __attribute__((aligned(16)));


struct __s128d {
  typedef float  vrs4_t __attribute__((vector_size(4 * sizeof(float))));
  typedef double vrd2_t __attribute__((vector_size(2 * sizeof(double))));
  union {
    float         xf[4];
    int           xi[4];
    unsigned int  xui[4];
    unsigned long xul[2];
    double        xd[2];
    vrs4_t        xfrd;
    vrd2_t        xdrd;
    float32x4_t   nfrd;
    float64x2_t   ndrd;
    long double   b;
    unsigned char c[16];
  } __attribute__((aligned(16)));


  __s128d() : b(0) { }

  __s128d(const double d[2]) : b(0) {
    xdrd[0] = d[0];
    xdrd[1] = d[1];
  }

  __s128d(const float f[2]) : b(0) {
    xdrd[0] = f[0];
    xdrd[1] = f[1];
  }

  __s128d(const int i[2]) : b(0) {
    xdrd[0] = i[0];
    xdrd[1] = i[1];
  }

  __s128d(const unsigned int i[2]) : b(0) {
    xdrd[0] = i[0];
    xdrd[1] = i[1];
  }

  __s128d(float f0, float f1) : b(0) {
    xdrd[0] = f0;
    xdrd[1] = f1;
  }

  __s128d(double d0, double d1) : b(0) {
    xdrd[0] = d0;
    xdrd[1] = d1;
  }

  __s128d(const long double& v) : b(v) { }

  __s128d(const __s128d& rhs) : b(rhs.b) { }

  __s128d(const vrd2_t& v) : xdrd(v) { }

  __s128d(const vrs4_t& v) : xfrd(v) { }

  inline __s128d& operator=(const __s128d& rhs) {
    if (this != &rhs)
      xdrd = rhs.xdrd;

    return *this;
  }

  inline __s128d& operator=(long double ld) {
    b = ld;
    return *this;
  }

  inline __s128d& operator=(const vrd2_t& rhs) {
    xdrd = rhs;
    return *this;
  }

  inline __s128d& operator=(const vrs4_t& rhs) {
    xfrd = rhs;
    return *this;
  }

  inline operator bool() const {
    return xdrd[0] != 0.00 && xdrd[1] != 0.00;
  }

  inline operator long double() const {
    return b;
  }

  inline operator vrs4_t() const {
    return xfrd;
  }

  inline operator vrd2_t() const {
    return xdrd;
  }

  inline __s128d operator+(unsigned int i) const {
    __s128d r(*this);

    if (i != 0U) {
      r.xdrd[0] += i;
      r.xdrd[1] += i;
    }

    return r;
  }

  inline __s128d operator+(int i) const {
    __s128d r(*this);

    if (i != 0) {
      r.xdrd[0] += i;
      r.xdrd[1] += i;
    }

    return r;
  }

  inline __s128d operator+(float f) const {
    __s128d r(*this);

    if (f != 0.0) {
      r.xdrd[0] += f;
      r.xdrd[1] += f;
    }

    return r;
  }

  inline __s128d operator+(double d) const {
    __s128d r(*this);

    if (d != 0.0) {
      r.xdrd[0] += d;
      r.xdrd[1] += d;
    }

    return r;
  }

  inline __s128d operator+(const long double& ld) const {
    __s128d r(*this);

    if (ld != 0.0)
      r.b += ld;

    return r;
  }

  inline __s128d operator+(const __s128d& s) const {
    __s128d r(*this);

    r.xdrd[0] += s.xdrd[0];
    r.xdrd[1] += s.xdrd[1];

    return r;
  }

  inline __s128d operator-(const __s128d& s) const {
    __s128d r(*this);

    r.xdrd[0] -= s.xdrd[0];
    r.xdrd[1] -= s.xdrd[1];

    return r;
  }

  inline __s128d operator-() const {
    return __s128d(-xdrd[0], -xdrd[1]);
  }

  inline __s128d operator*(const __s128d& rhs) const {
    __s128d r(*this);

    r.xdrd[0] *= rhs.xdrd[0];
    r.xdrd[1] *= rhs.xdrd[1];

    return r;
  }

  inline __s128d operator/(const __s128d& rhs) const {
    __s128d r(*this);

    r.xdrd[0] /= rhs.xdrd[0];
    r.xdrd[1] /= rhs.xdrd[1];

    return r;
  }

  inline __s128d& operator+=(unsigned int i) {
    if (i != 0U) {
      xdrd[0] += i;
      xdrd[1] += i;
    }

    return *this;
  }

  inline __s128d& operator+=(int i) {
    if (i != 0) {
      xdrd[0] += i;
      xdrd[1] += i;
    }

    return *this;
  }

  inline __s128d& operator+=(float f) {
    if (f != 0.0) {
      xdrd[0] += f;
      xdrd[1] += f;
    }

    return *this;
  }

  inline __s128d& operator+=(double d) {
    if (d != 0.0) {
      xdrd[0] += d;
      xdrd[1] += d;
    }

    return *this;
  }

  inline __s128d& operator+=(long double ld) {
    if (ld != 0.0) {
      xdrd[0] += (double) ld;
      xdrd[1] += (double) ld;
    }

    return *this;
  }

  inline __s128d& operator+=(const __s128d& rhs) {
    xdrd[0] += rhs.xdrd[0];
    xdrd[1] += rhs.xdrd[1];

    return *this;
  }

  inline __s128d& operator*=(const __s128d& rhs) {
    xdrd[0] *= rhs.xdrd[0];
    xdrd[1] *= rhs.xdrd[1];

    return *this;
  }

  inline __s128d& operator/=(const __s128d& rhs) {
    xdrd[0] /= rhs.xdrd[0];
    xdrd[1] /= rhs.xdrd[1];

    return *this;
  }

  inline __s128d& operator-=(const __s128d& rhs) {
    xdrd[0] -= rhs.xdrd[0];
    xdrd[1] -= rhs.xdrd[1];

    return *this;
  }
} __attribute__((aligned(16)));

struct __s128i {
  typedef enum __s128iType {
    Invalid = 0,
    SignedInt = 1,
    UnsignedInt = 2,
    SignedLong = 3,
    UnsignedLong = 4,
  } s128iType;

  typedef int32_t vis4_t __attribute__((vector_size(4 * sizeof(int32_t))));
  typedef int64_t vid2_t __attribute__((vector_size(2 * sizeof(int64_t))));

  union {
    float           xf[4];
    int             xi[4];
    unsigned int    xui[4];
    long            xl[2];
    unsigned long   xul[2];
    vis4_t          xird;
    vid2_t          xlrd;
    int32x4_t       nird;
    int64x2_t       nlrd;
    double          xd[2];
    long double     b;
    unsigned char   c[16];
  } __attribute__((aligned(16)));

  s128iType type;

  __s128i() : b(0), type(UnsignedInt) { }

  __s128i(const float f[4]) : b(0), type(SignedInt) {
    xird[0] = f[0];
    xird[1] = f[1];
    xird[2] = f[2];
    xird[3] = f[3];
  }

  __s128i(const double d[2]) : b(0), type(SignedLong) {
    xlrd[0] = d[0];
    xlrd[1] = d[1];
  }

  __s128i(const int i[4]) : b(0), type(SignedInt) {
    xird[0] = i[0];
    xird[1] = i[1];
    xird[2] = i[2];
    xird[3] = i[3];
  }

  __s128i(const unsigned int i[4]) : b(0), type(UnsignedInt) {
    xui[0] = i[0];
    xui[1] = i[1];
    xui[2] = i[2];
    xui[3] = i[3];
  }

  __s128i(int i0, int i1, int i2, int i3) : b(0), type(SignedInt) {
    xird[0] = i0;
    xird[1] = i1;
    xird[2] = i2;
    xird[3] = i3;
  }

  __s128i(unsigned int i0, unsigned int i1,
          unsigned int i2, unsigned int i3)
  : b(0), type(UnsignedInt) {
    xui[0] = i0;
    xui[1] = i1;
    xui[2] = i2;
    xui[3] = i3;
  }

  __s128i(long l0, long l1)
  : b(0), type(SignedLong) {
    xlrd[0] = l0;
    xlrd[1] = l1;
  }

  __s128i(unsigned long l0, unsigned long l1)
  : b(0), type(UnsignedLong) {
    xul[0] = l0;
    xul[1] = l1;
  }

  __s128i(const vis4_t& v) : xird(v), type(SignedInt) { }

  __s128i(const vid2_t& v) : xlrd(v), type(SignedLong) { }

  __s128i(const long double& v) : b(v), type(UnsignedInt) { }

  __s128i(const __s128i& rhs) : b(rhs.b), type(rhs.type) { }

  inline __s128i& operator=(const __s128i& rhs) {
    if (this != &rhs) {
      b = rhs.b;
      type = rhs.type;
    }

    return *this;
  }

  inline __s128i& operator=(const long double& ld) {
    b = ld;
    type = UnsignedInt;
    return *this;
  }

  inline __s128i& operator=(int i) {
    xird[0] = i;
    xird[1] = i;
    xird[2] = i;
    xird[3] = i;
    type = SignedInt;
    return *this;
  }

  inline __s128i& operator=(unsigned int i) {
    xird[0] = i;
    xird[1] = i;
    xird[2] = i;
    xird[3] = i;
    type = UnsignedInt;
    return *this;
  }

  inline __s128i& operator=(const long& l) {
    xlrd[0] = l;
    xlrd[1] = l;
    type = SignedLong;
    return *this;
  }

  inline __s128i& operator=(const unsigned long& l) {
    xul[0] = l;
    xul[1] = l;
    type = SignedLong;
    return *this;
  }

  inline __s128i& operator=(const vis4_t& rhs) {
    xird = rhs;
    return *this;
  }

  inline __s128i& operator=(const vid2_t& rhs) {
    xlrd = rhs;
    return *this;
  }

  inline operator bool() const {
    return xird[0] == 0U && xird[1] == 0U && xird[2] == 0U && xird[3] == 0U;
  }

  inline operator long double() const {
    return b;
  }

  inline operator vis4_t() const {
    return xird;
  }

  inline operator vid2_t() const {
    return xlrd;
  }

#if __GNUC__ < 8
  inline operator int32x4_t() const {
    return nird;
  }

  inline operator int64x2_t() const {
    return nlrd;
  }
#endif

  inline __s128i operator+(unsigned int i) const {
    __s128i r(*this);

    if (i != 0U) {
      r.xird[0] += i;
      r.xird[1] += i;
      r.xird[2] += i;
      r.xird[3] += i;
    }

    return r;
  }

  inline __s128i operator+(int i) const {
    __s128i r(*this);

    if (i != 0) {
      r.xird[0] += i;
      r.xird[1] += i;
      r.xird[2] += i;
      r.xird[3] += i;
    }

    return r;
  }

  inline __s128i operator+(long l) const {
    __s128i r(*this);

    if (l != 0L) {
      r.xlrd[0] += l;
      r.xlrd[1] += l;
    }

    return r;
  }

  inline __s128i operator+(unsigned long l) const {
    __s128i r(*this);

    if (l != 0UL) {
      r.xul[0] += l;
      r.xul[1] += l;
    }

    return r;
  }

  inline __s128i operator+(float f) const {
    __s128i r(*this);

    if (f != 0.0) {
      r.xird[0] += f;
      r.xird[1] += f;
      r.xird[2] += f;
      r.xird[3] += f;
    }

    return r;
  }

  inline __s128i operator+(double d) const {
    __s128i r(*this);

    if (d != 0.0) {
      r.xird[0] += d;
      r.xird[1] += d;
      r.xird[2] += d;
      r.xird[3] += d;
    }

    return r;
  }

  inline __s128f operator/(const __s128i& rhs) const {
    __s128f r;

    if (type == rhs.type && type == SignedInt) {
      r.xfrd[0] = (float) xird[0] / rhs.xird[0];
      r.xfrd[1] = (float) xird[1] / rhs.xird[1];
      r.xfrd[2] = (float) xird[2] / rhs.xird[2];
      r.xfrd[3] = (float) xird[3] / rhs.xird[3];
    } else if (type == rhs.type && type == UnsignedInt) {
      r.xfrd[0] = (float) xui[0] / rhs.xui[0];
      r.xfrd[1] = (float) xui[1] / rhs.xui[1];
      r.xfrd[2] = (float) xui[2] / rhs.xui[2];
      r.xfrd[3] = (float) xui[3] / rhs.xui[3];
    }

    return r;
  }

  inline __s128f operator/(int i) const {
    __s128f r;

    r.xfrd[0] = (float) xird[0] / i;
    r.xfrd[1] = (float) xird[1] / i;
    r.xfrd[2] = (float) xird[2] / i;
    r.xfrd[3] = (float) xird[3] / i;

    return r;
  }

  inline __s128f operator/(unsigned int i) const {
    __s128f r;

    r.xfrd[0] = (float) xui[0] / i;
    r.xfrd[1] = (float) xui[1] / i;
    r.xfrd[2] = (float) xui[2] / i;
    r.xfrd[3] = (float) xui[3] / i;

    return r;
  }

  inline __s128i operator+(const __s128i& s) const {
    __s128i r;
    r = xird + s.xird;
    return r;
  }

  inline __s128i operator-(const __s128i& s) const {
    __s128i r;
    r = xird - s.xird;
    return r;
  }

  inline __s128i operator_() const {
    return __s128i(-xird[0], -xird[1], -xird[2], xird[3]);
  }

  inline __s128i operator+(const long double& ld) const {
    __s128i r(*this);

    if (ld != 0.0)
      r.b += ld;

    return r;
  }

  inline __s128i& operator+=(unsigned int i) {
    if (i != 0U)
      xird = xird + static_cast<int32_t>(i);

    return *this;
  }

  inline __s128i& operator+=(int i) {
    if (i != 0)
      xird = xird + static_cast<int32_t>(i);

    return *this;
  }

  inline __s128i& operator+=(long l) {
    if (l != 0U) {
      xlrd[0] += l;
      xlrd[1] += l;
    }

    return *this;
  }

  inline __s128i& operator+=(unsigned long l) {
    if (l != 0UL)
      xlrd = xlrd + static_cast<int64_t>(l);

    return *this;
  }

  inline __s128i& operator+=(float f) {
    if (f != 0.0) {
      xird[0] += f;
      xird[1] += f;
      xird[2] += f;
      xird[3] += f;
    }

    return *this;
  }

  inline __s128i& operator+=(double d) {
    if (d != 0.0) {
      xird[0] += d;
      xird[1] += d;
      xird[2] += d;
      xird[3] += d;
    }

    return *this;
  }

  inline __s128i& operator+=(const long double& ld) {
    if (ld != 0.0) {
      xird[0] += ld;
      xird[1] += ld;
      xird[2] += ld;
      xird[3] += ld;
    }

    return *this;
  }

  inline __s128i& operator+=(const __s128i& rhs) {
    xird[0] += rhs.xird[0];
    xird[1] += rhs.xird[1];
    xird[2] += rhs.xird[2];
    xird[3] += rhs.xird[3];

    return *this;
  }

  inline __s128i& operator-=(const __s128i& rhs) {
    xird[0] -= rhs.xird[0];
    xird[1] -= rhs.xird[1];
    xird[2] -= rhs.xird[2];
    xird[3] -= rhs.xird[3];

    return *this;
  }
} __attribute__((aligned(16)));

typedef struct __s128f __m128;
typedef struct __s128i __m128i;
typedef struct __s128d __m128d;

static inline void
__attribute__((always_inline))
vec_st(const __m128& vld, int v, unsigned char* t)
{
  __m128 vldx = v ? vld + v : vld;

  t[0] = vldx.c[0];
  t[1] = vldx.c[1];
  t[2] = vldx.c[2];
  t[3] = vldx.c[3];
  t[4] = vldx.c[4];
  t[5] = vldx.c[5];
  t[6] = vldx.c[6];
  t[7] = vldx.c[7];
  t[8] = vldx.c[8];
  t[9] = vldx.c[9];
  t[10] = vldx.c[10];
  t[11] = vldx.c[11];
  t[12] = vldx.c[12];
  t[13] = vldx.c[13];
  t[14] = vldx.c[14];
  t[15] = vldx.c[15];
}

static inline void
__attribute__((always_inline))
vec_st(const __m128& vld, int v, unsigned int* t)
{
  __m128 vldx = v ? vld + v : vld;

  t[0] = vldx.xui[0];
  t[1] = vldx.xui[1];
  t[2] = vldx.xui[2];
  t[3] = vldx.xui[3];
}

static inline void
__attribute__((always_inline))
vec_st(const __m128i& vldi, int v, unsigned char* t)
{
  __m128i vldx = v ? vldi + v : vldi;

  t[0] = vldx.c[0];
  t[1] = vldx.c[1];
  t[2] = vldx.c[2];
  t[3] = vldx.c[3];
  t[4] = vldx.c[4];
  t[5] = vldx.c[5];
  t[6] = vldx.c[6];
  t[7] = vldx.c[7];
  t[8] = vldx.c[8];
  t[9] = vldx.c[9];
  t[10] = vldx.c[10];
  t[11] = vldx.c[11];
  t[12] = vldx.c[12];
  t[13] = vldx.c[13];
  t[14] = vldx.c[14];
  t[15] = vldx.c[15];
}

static inline void
__attribute__((always_inline))
vec_st(const __m128i& vldi, int v, unsigned int* t)
{
  __m128i vldx = v ? vldi + v : vldi;

  t[0] = vldx.xui[0];
  t[1] = vldx.xui[1];
  t[2] = vldx.xui[2];
  t[3] = vldx.xui[3];
}

static inline void
__attribute__((always_inline))
vec_st(const __m128d& vldd, int v, unsigned char* t)
{
  __m128d vldx = v ? vldd + v : vldd;

  t[0] = vldx.c[0];
  t[1] = vldx.c[1];
  t[2] = vldx.c[2];
  t[3] = vldx.c[3];
  t[4] = vldx.c[4];
  t[5] = vldx.c[5];
  t[6] = vldx.c[6];
  t[7] = vldx.c[7];
  t[8] = vldx.c[8];
  t[9] = vldx.c[9];
  t[10] = vldx.c[10];
  t[11] = vldx.c[11];
  t[12] = vldx.c[12];
  t[13] = vldx.c[13];
  t[14] = vldx.c[14];
  t[15] = vldx.c[15];
}

static inline void
__attribute__((always_inline))
vec_st(const __m128d& vldd, int v, unsigned int* t)
{
  __m128d vldx = v ? vldd + v : vldd;

  t[0] = vldx.xui[0];
  t[1] = vldx.xui[1];
  t[2] = vldx.xui[2];
  t[3] = vldx.xui[3];
}

static inline unsigned int
__attribute__((always_inline))
vec_extract(const __m128& vlf, unsigned int i)
{
  return static_cast<unsigned int>(vlf.xi[i]);
}

static inline unsigned int
__attribute__((always_inline))
vec_extract(const __m128i& vli, unsigned int i)
{
  return static_cast<unsigned int>(vli.xi[i]);
}

static inline unsigned int
__attribute__((always_inline))
vec_extract(const __m128d& vld, unsigned int i)
{
  return static_cast<unsigned int>(vld.xi[i]);
}

static inline __m128i
__attribute__((always_inline))
vec_insert(unsigned int v, const __m128i& vld, unsigned int i)
{
  __m128i vldx = vld;
  vldx.xird[i] = v;
  return vldx;
}

static inline __m128i
__attribute__((always_inline))
vec_insert(int v, const __m128i& vld, unsigned int i)
{
  __m128i vldx = vld;
  vldx.xird[i] = v;
  return vldx;
}

static inline __m128
__attribute__((always_inline))
vec_insert(float v, const __m128& vld, unsigned int i)
{
  __m128 vldx = vld;
  vldx.xfrd[i] = v;
  return vldx;
}

static inline __m128d
__attribute__((always_inline))
vec_insert(double v, const __m128d& vld, unsigned int i)
{
  __m128d vldx = vld;
  vldx.xdrd[i] = v;
  return vldx;
}

static inline __m128i
__attribute__((always_inline))
vec_splats(unsigned int i)
{
  __m128i r(i, i, i, i);
  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_splats(int i)
{
  __m128i r(i, i, i, i);
  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_splats(long int i)
{
  __m128i r;
  r.xl[0] = r.xi[2] = (long int) ((i & 0xFFFFFFFF00000000LL) >> 32);
  r.xl[1] = r.xi[3] = (long int) (i & 0xFFFFFFFFLL);
  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_splats(unsigned long i)
{
  __m128i r;
  r.xl[0] = r.xi[2] = (unsigned long) ((i & 0xFFFFFFFF00000000ULL) >> 32);
  r.xl[1] = r.xi[3] = (unsigned long) (i & 0xFFFFFFFFULL);
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_splats(float f)
{
  __m128 r(f, f, f, f);
  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_splats(double d)
{
  __m128d r(d, d);
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_sqrt(const __m128& a)
{
  __m128 r;

  r.xfrd[0] = sqrtf(a.xfrd[0]);
  r.xfrd[1] = sqrtf(a.xfrd[1]);
  r.xfrd[2] = sqrtf(a.xfrd[2]);
  r.xfrd[3] = sqrtf(a.xfrd[3]);

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_sqrt(const __m128d& a)
{
  __m128d r;

  r.xdrd[0] = sqrt(a.xdrd[0]);
  r.xdrd[1] = sqrt(a.xdrd[1]);

  return r;

}

static inline __m128i
__attribute__((always_inline))
vec_ld(unsigned int v, __m128i* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128i
__attribute__((always_inline))
vec_ld(int v, __m128i* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128i
__attribute__((always_inline))
vec_ld(unsigned long v, __m128i* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128i
__attribute__((always_inline))
vec_ld(long v, __m128i* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128
__attribute__((always_inline))
vec_ld(unsigned int v, __m128* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128
__attribute__((always_inline))
vec_ld(int v, __m128* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128::vrd2_t
__attribute__((always_inline))
vec_ld(int v, float vld[4])
{
  __m128 r(vld);
  r += v;
  return r.operator __m128::vrd2_t();
}

static inline __m128::vrd2_t
__attribute__((always_inline))
vec_ld(unsigned int v, float vld[4])
{
  __m128 r(vld);
  r += v;
  return r.operator __m128::vrd2_t();
}

static inline __m128d
__attribute__((always_inline))
vec_ld(unsigned int v, __m128d* vld)
{
  *vld += v;
  return *vld;
}

static inline __m128d
__attribute__((always_inline))
vec_ld(int v, __m128d* vld)
{
  *vld += v;
  return *vld;
}

static inline int
__attribute__((always_inline))
vec_any_ne(const __m128i& a, const __m128i& b)
{
  for (unsigned i = 0; i < 4; ++i) {
    if (a.xird[i] != b.xird[i])
      return 1;
  }

  return 0;
}

static inline int
__attribute__((always_inline))
vec_any_ne(const __m128& a, const __m128& b)
{
  for (unsigned i = 0; i < 4; ++i) {
    if (a.xfrd[i] != b.xfrd[i])
      return 1;
  }

  return 0;
}

static inline int
__attribute__((always_inline))
vec_any_ne(const __m128d& a, const __m128d& b)
{
  for (unsigned i = 0; i < 2; ++i) {
    if (a.xdrd[i] != b.xdrd[i])
      return 1;
  }

  return 0;
}

static inline __m128
__attribute__((always_inline))
vec_add(const __m128& a, const __m128& b)
{
  __m128 r;
  r.xfrd = a.xfrd + b.xfrd;
  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_add(const __m128i& a, const __m128i& b)
{
  __m128i r;
  r.xird = a.xird + b.xird;
  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_add(const __m128d& a, const __m128d& b)
{
  __m128d r;
  r.xdrd = a.xdrd + b.xdrd;
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_sub(const __m128& a, const __m128& b)
{
  __m128 r;
  r.xfrd = a.xfrd - b.xfrd;
  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_sub(const __m128i& a, const __m128i& b)
{
  __m128i r;
  r.xird = a.xird - b.xird;
  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_sub(const __m128d& a, const __m128d& b)
{
  __m128d r;
  r.xdrd = a.xdrd - b.xdrd;
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_madd(const __m128& a, const __m128& b, const __m128& c)
{
  __m128 r;
  r.xfrd = c.xfrd + a.xfrd * b.xfrd;
  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_madd(const __m128d& a, const __m128d& b, const __m128d& c)
{
  __m128d r;
  r.xdrd = c.xdrd + a.xdrd * b.xdrd;
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_msub(const __m128& a, const __m128& b, const __m128& c)
{
  __m128 r;
  r.xfrd = a.xfrd * b.xfrd - c.xfrd;
  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_msub(const __m128d& a, const __m128d& b, const __m128d& c)
{
  __m128d r;
  r.xdrd = a.xdrd * b.xdrd - c.xdrd;
  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_msub(const __m128i& a, const __m128i& b, const __m128i& c)
{
  __m128i r;
  r.xird = a.xird * b.xird - c.xird;
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_mul(const __m128& a, const __m128& b)
{
  return vec_madd(a, b, vec_splats(float(0.0)));
}

static inline __m128d
__attribute__((always_inline))
vec_mul(const __m128d& a, const __m128d& b)
{
  return vec_madd(a, b, vec_splats(double(0.0)));
}

static inline __m128
__attribute__((always_inline))
vec_cmpge(const __m128& a, const __m128& b)
{
  __m128 r;

  if (a.xfrd[0] >= b.xfrd[0])
    r.xi[0] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[1] >= b.xfrd[1])
    r.xi[1] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[2] >= b.xfrd[2])
    r.xi[2] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[3] >= b.xfrd[3])
    r.xi[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cmpge(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if (a.xird[0] >= b.xird[0])
    r.xird[0] = (unsigned) 0xFFFFFFFF;

  if (a.xird[1] >= b.xird[1])
    r.xird[1] = (unsigned) 0xFFFFFFFF;

  if (a.xird[2] >= b.xird[2])
    r.xird[2] = (unsigned) 0xFFFFFFFF;

  if (a.xird[3] >= b.xird[3])
    r.xird[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_cmpge(const __m128d& a, const __m128d& b)
{
  __m128d r;

  if (a.xdrd[0] >= b.xdrd[0]) {
    r.xui[0] = (unsigned) 0xFFFFFFFF;
    r.xui[1] = (unsigned) 0xFFFFFFFF;
  }

  if (a.xdrd[1] >= b.xdrd[1]) {
    r.xui[2] = (unsigned) 0xFFFFFFFF;
    r.xui[3] = (unsigned) 0xFFFFFFFF;
  }

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_cmpeq(const __m128& a, const __m128& b)
{
  __m128 r;

  if (a.xfrd[0] == b.xfrd[0])
    r.xui[0] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[1] == b.xfrd[1])
    r.xui[1] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[2] == b.xfrd[2])
    r.xui[2] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[3] == b.xfrd[3])
    r.xui[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_cmpeq(const __m128d& a, const __m128d& b)
{
  __m128d r;

  if (a.xdrd[0] == b.xdrd[0]) {
    r.xui[0] = (unsigned) 0xFFFFFFFF;
    r.xui[1] = (unsigned) 0xFFFFFFFF;
  }

  if (a.xdrd[1] == b.xdrd[1]) {
    r.xui[2] = (unsigned) 0xFFFFFFFF;
    r.xui[3] = (unsigned) 0xFFFFFFFF;
  }

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cmpeq(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if (a.xird[0] == b.xird[0])
    r.xird[0] = (unsigned) 0xFFFFFFFF;

  if (a.xird[1] == b.xird[1])
    r.xird[1] = (unsigned) 0xFFFFFFFF;

  if (a.xird[2] == b.xird[2])
    r.xird[2] = (unsigned) 0xFFFFFFFF;

  if (a.xird[3] == b.xird[3])
    r.xird[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_cmple(const __m128& a, const __m128& b)
{
  __m128 r;

  if (a.xfrd[0] <= b.xfrd[0])
    r.xui[0] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[1] <= b.xfrd[1])
    r.xui[0] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[2] <= b.xfrd[2])
    r.xui[2] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[3] <= b.xfrd[3])
    r.xui[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_cmple(const __m128d& a, const __m128d& b)
{
  __m128d r;

  if (a.xdrd[0] <= b.xdrd[0]) {
    r.xui[0] = (unsigned) 0xFFFFFFFF;
    r.xui[1] = (unsigned) 0xFFFFFFFF;
  }

  if (a.xdrd[1] <= b.xdrd[1]) {
    r.xui[2] = (unsigned) 0xFFFFFFFF;
    r.xui[3] = (unsigned) 0xFFFFFFFF;
  }

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cmple(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if (a.xird[0] <= b.xird[0])
    r.xui[0] = (unsigned) 0xFFFFFFFF;

  if (a.xird[1] <= b.xird[1])
    r.xui[1] = (unsigned) 0xFFFFFFFF;

  if (a.xird[2] <= b.xird[2])
    r.xui[2] = (unsigned) 0xFFFFFFFF;

  if (a.xird[3] <= b.xird[3])
    r.xui[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_cmpgt(const __m128& a, const __m128& b)
{
  __m128 r;

  if (a.xfrd[0] > b.xfrd[0])
    r.xui[0] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[1] > b.xfrd[1])
    r.xui[1] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[2] > b.xfrd[2])
    r.xui[2] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[3] > b.xfrd[3])
    r.xui[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_cmpgt(const __m128d& a, const __m128d& b)
{
  __m128d r;

  if (a.xdrd[0] > b.xdrd[0]) {
    r.xui[0] = (unsigned) 0xFFFFFFFF;
    r.xui[1] = (unsigned) 0xFFFFFFFF;
  }

  if (a.xdrd[1] > b.xdrd[1]) {
    r.xui[2] = (unsigned) 0xFFFFFFFF;
    r.xui[3] = (unsigned) 0xFFFFFFFF;
  }

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cmpgt(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if (a.xird[0] > b.xird[0])
    r.xui[0] = (unsigned) 0xFFFFFFFF;

  if (a.xird[1] > b.xird[1])
    r.xui[1] = (unsigned) 0xFFFFFFFF;

  if (a.xird[2] > b.xird[2])
    r.xui[2] = (unsigned) 0xFFFFFFFF;

  if (a.xird[3] > b.xird[3])
    r.xui[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_and(const __m128& a, const __m128& b)
{
  __m128 r(a);

  r.c[0] &= b.c[0];
  r.c[1] &= b.c[1];
  r.c[2] &= b.c[2];
  r.c[3] &= b.c[3];
  r.c[4] &= b.c[4];
  r.c[5] &= b.c[5];
  r.c[6] &= b.c[6];
  r.c[7] &= b.c[7];
  r.c[8] &= b.c[8];
  r.c[9] &= b.c[9];
  r.c[10] &= b.c[10];
  r.c[11] &= b.c[11];
  r.c[12] &= b.c[12];
  r.c[13] &= b.c[13];
  r.c[14] &= b.c[14];
  r.c[15] &= b.c[15];

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_and(const __m128d& a, const __m128d& b)
{
  __m128d r(a);

  r.c[0] &= b.c[0];
  r.c[1] &= b.c[1];
  r.c[2] &= b.c[2];
  r.c[3] &= b.c[3];
  r.c[4] &= b.c[4];
  r.c[5] &= b.c[5];
  r.c[6] &= b.c[6];
  r.c[7] &= b.c[7];
  r.c[8] &= b.c[8];
  r.c[9] &= b.c[9];
  r.c[10] &= b.c[10];
  r.c[11] &= b.c[11];
  r.c[12] &= b.c[12];
  r.c[13] &= b.c[13];
  r.c[14] &= b.c[14];
  r.c[15] &= b.c[15];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_and(const __m128i& a, const __m128i& b)
{
  __m128i r(a);

  r.xird[0] &= b.xird[0];
  r.xird[1] &= b.xird[1];
  r.xird[2] &= b.xird[2];
  r.xird[3] &= b.xird[3];

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_xor(const __m128& a, const __m128& b)
{
  __m128 r(a);

  r.c[0] ^= b.c[0];
  r.c[1] ^= b.c[1];
  r.c[2] ^= b.c[2];
  r.c[3] ^= b.c[3];
  r.c[4] ^= b.c[4];
  r.c[5] ^= b.c[5];
  r.c[6] ^= b.c[6];
  r.c[7] ^= b.c[7];
  r.c[8] ^= b.c[8];
  r.c[9] ^= b.c[9];
  r.c[10] ^= b.c[10];
  r.c[11] ^= b.c[11];
  r.c[12] ^= b.c[12];
  r.c[13] ^= b.c[13];
  r.c[14] ^= b.c[14];
  r.c[15] ^= b.c[15];

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_xor(const __m128d& a, const __m128d& b)
{
  __m128d r(a);

  r.c[0] ^= b.c[0];
  r.c[1] ^= b.c[1];
  r.c[2] ^= b.c[2];
  r.c[3] ^= b.c[3];
  r.c[4] ^= b.c[4];
  r.c[5] ^= b.c[5];
  r.c[6] ^= b.c[6];
  r.c[7] ^= b.c[7];
  r.c[8] ^= b.c[8];
  r.c[9] ^= b.c[9];
  r.c[10] ^= b.c[10];
  r.c[11] ^= b.c[11];
  r.c[12] ^= b.c[12];
  r.c[13] ^= b.c[13];
  r.c[14] ^= b.c[14];
  r.c[15] ^= b.c[15];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_xor(const __m128i& a, const __m128i& b)
{
  __m128i r(a);

  r.xird[0] ^= b.xird[0];
  r.xird[1] ^= b.xird[1];
  r.xird[2] ^= b.xird[2];
  r.xird[3] ^= b.xird[3];

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_or(const __m128& a, const __m128& b)
{
  __m128 r(a);

  r.c[0] |= b.c[0];
  r.c[1] |= b.c[1];
  r.c[2] |= b.c[2];
  r.c[3] |= b.c[3];
  r.c[4] |= b.c[4];
  r.c[5] |= b.c[5];
  r.c[6] |= b.c[6];
  r.c[7] |= b.c[7];
  r.c[8] |= b.c[8];
  r.c[9] |= b.c[9];
  r.c[10] |= b.c[10];
  r.c[11] |= b.c[11];
  r.c[12] |= b.c[12];
  r.c[13] |= b.c[13];
  r.c[14] |= b.c[14];
  r.c[15] |= b.c[15];

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_or(const __m128d& a, const __m128d& b)
{
  __m128d r(a);

  r.c[0] |= b.c[0];
  r.c[1] |= b.c[1];
  r.c[2] |= b.c[2];
  r.c[3] |= b.c[3];
  r.c[4] |= b.c[4];
  r.c[5] |= b.c[5];
  r.c[6] |= b.c[6];
  r.c[7] |= b.c[7];
  r.c[8] |= b.c[8];
  r.c[9] |= b.c[9];
  r.c[10] |= b.c[10];
  r.c[11] |= b.c[11];
  r.c[12] |= b.c[12];
  r.c[13] |= b.c[13];
  r.c[14] |= b.c[14];
  r.c[15] |= b.c[15];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_or(const __m128i& a, const __m128i& b)
{
  __m128i r(a);

  r.xird[0] |= b.xird[0];
  r.xird[1] |= b.xird[1];
  r.xird[2] |= b.xird[2];
  r.xird[3] |= b.xird[3];

  return r;
}

static inline
__attribute__((always_inline))
__m128 vec_compl(const __m128& a)
{
  __m128 r(a);

  r.xi[0] = ~r.xi[0];
  r.xi[1] = ~r.xi[1];
  r.xi[2] = ~r.xi[2];
  r.xi[3] = ~r.xi[3];

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_compl(const __m128d& a)
{
  __m128d r(a);

  r.xi[0] = ~r.xi[0];
  r.xi[1] = ~r.xi[1];
  r.xi[2] = ~r.xi[2];
  r.xi[3] = ~r.xi[3];

  return r;
}

static inline
__attribute__((always_inline))
__m128i vec_compl(const __m128i& a)
{
  __m128i r(a);

  r.xi[0] = ~r.xi[0];
  r.xi[1] = ~r.xi[1];
  r.xi[2] = ~r.xi[2];
  r.xi[3] = ~r.xi[3];

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_andc(const __m128& a, const __m128& b)
{
  return vec_and(a, vec_compl(b));
}

static inline __m128d
__attribute__((always_inline))
vec_andc(const __m128d& a, const __m128d& b)
{
  return vec_and(a, vec_compl(b));
}

static inline __m128i
__attribute__((always_inline))
vec_andc(const __m128i& a, const __m128i& b)
{
  return vec_and(a, vec_compl(b));
}

static inline __m128i
__attribute__((always_inline))
vec_sl(const __m128i& a, const __m128i& b)
{
  __m128i r;

  r.xird[0] = (a.xird[0] << b.xird[0]) % 32;
  r.xird[1] = (a.xird[1] << b.xird[1]) % 32;
  r.xird[2] = (a.xird[2] << b.xird[2]) % 32;
  r.xird[3] = (a.xird[3] << b.xird[3]) % 32;

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_sr(const __m128i& a, const __m128i& b)
{
  __m128i r;

  r.xird[0] = (a.xird[0] >> b.xird[0]) % 32;
  r.xird[1] = (a.xird[1] >> b.xird[1]) % 32;
  r.xird[2] = (a.xird[2] >> b.xird[2]) % 32;
  r.xird[3] = (a.xird[3] >> b.xird[3]) % 32;

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_ctf(const __m128i& a, unsigned i)
{
  __m128 r;
  assert(i < 32U && "Invalid exponent!");

  if (a.type == __m128i::SignedInt) {
    r.xfrd[0] = (float) ((a.xi[0] + ((a.xi[0] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[1] = (float) ((a.xi[1] + ((a.xi[1] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[2] = (float) ((a.xi[2] + ((a.xi[2] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[3] = (float) ((a.xi[3] + ((a.xi[3] >> 31) & ((1 << i) + ~0))) >> i);
  } else if (a.type == __m128i::UnsignedInt) {
    r.xfrd[0] =
      (float) ((a.xui[0] + ((a.xui[0] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[1] =
      (float) ((a.xui[1] + ((a.xui[1] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[2] =
      (float) ((a.xui[2] + ((a.xui[2] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[3] =
      (float) ((a.xui[3] + ((a.xui[3] >> 31) & ((1 << i) + ~0))) >> i);
  } else if (a.type == __m128i::SignedLong) {
    r.xfrd[0] =
      (float) ((a.xl[0] + ((a.xl[0] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[2] =
      (float) ((a.xl[1] + ((a.xl[1] >> 31) & ((1 << i) + ~0))) >> i);
  } else if (a.type == __m128i::UnsignedLong) {
    r.xfrd[0] =
      (float) ((a.xul[0] + ((a.xul[0] >> 31) & ((1 << i) + ~0))) >> i);
    r.xfrd[2] =
      (float) ((a.xul[1] + ((a.xul[1] >> 31) & ((1 << i) + ~0))) >> i);
  }

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_fixed(const __m128& a)
{
  __m128i r;

  r.xird[0] = (int) a.xfrd[0];
  r.xird[1] = (int) a.xfrd[1];
  r.xird[2] = (int) a.xfrd[2];
  r.xird[3] = (int) a.xfrd[3];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_fixed(const __m128d& a)
{
  __m128i r;

  r.xird[0] = (int) a.xdrd[0];
  r.xird[2] = (int) a.xdrd[1];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cts(const __m128& a, int i)
{
  __m128i r;

  assert((i >= 0 && i < 32) && "Invalid exponent!");

  // FIXME: Expand to bitwise.
  r.xird[0] = (int) ldexpf(a.xfrd[0], (int) i);
  r.xird[1] = (int) ldexpf(a.xfrd[1], (int) i);
  r.xird[2] = (int) ldexpf(a.xfrd[2], (int) i);
  r.xird[3] = (int) ldexpf(a.xfrd[3], (int) i);

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cts(const __m128d& a, int i)
{
  __m128i r;

  assert((i >= 0) && (i < 32) && "Invalid exponent!");

  // FIXME: Expand to bitwise.
  // The values of xi[1] and xi[3] are undefined (zero).
  r.xird[0] = (int) ldexp(a.xdrd[0], (int) i);
  r.xird[2] = (int) ldexp(a.xdrd[1], (int) i);

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_cmplt(const __m128& a, const __m128& b)
{
  __m128 r;

  if (a.xfrd[0] < b.xfrd[0])
    r.xi[0] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[1] < b.xfrd[1])
    r.xi[1] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[2] < b.xfrd[2])
    r.xi[2] = (unsigned) 0xFFFFFFFF;

  if (a.xfrd[3] < b.xfrd[3])
    r.xi[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_cmplt(const __m128d& a, const __m128d& b)
{
  __m128d r;

  if (a.xdrd[0] < b.xdrd[0]) {
    r.xi[0] = (unsigned) 0xFFFFFFFF;
    r.xi[1] = (unsigned) 0xFFFFFFFF;
  }

  if (a.xdrd[1] < b.xdrd[1]) {
    r.xi[2] = (unsigned) 0xFFFFFFFF;
    r.xi[3] = (unsigned) 0xFFFFFFFF;
  }

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_trunc(const __m128& a)
{
  __m128 r;

  r.xfrd[0] = (float) ((int) a.xfrd[0]);
  r.xfrd[1] = (float) ((int) a.xfrd[1]);
  r.xfrd[2] = (float) ((int) a.xfrd[2]);
  r.xfrd[3] = (float) ((int) a.xfrd[3]);

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_trunc(const __m128d& a)
{
  __m128d r;

  r.xdrd[0] = (double) ((long) a.xdrd[0]);
  r.xdrd[1] = (double) ((long) a.xdrd[1]);

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_floor(const __m128& a)
{
  __m128 r;

  r.xfrd[0] = (int) (a.xfrd[0] + 16777215.0f) - (int) 16777215;
  r.xfrd[1] = (int) (a.xfrd[1] + 16777215.0f) - (int) 16777215;
  r.xfrd[2] = (int) (a.xfrd[2] + 16777215.0f) - (int) 16777215;
  r.xfrd[3] = (int) (a.xfrd[3] + 16777215.0f) - (int) 16777215;

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_floor(const __m128d& a)
{
  __m128d r;

  r.xdrd[0] = (long) (a.xdrd[0] + 2147418111.00) - (long) 2147418111;
  r.xdrd[1] = (long) (a.xdrd[1] + 2147418111.00) - (long) 2147418111;

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_div(const __m128& a, const __m128& b)
{
  __m128 r = a / b;
  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_div(const __m128d& a, const __m128d& b)
{
  __m128d r = a / b;
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_div(const __m128i& a, const __m128i& b)
{
  __m128 r = a / b;
  return r;
}

static inline __m128
__attribute__((always_inline))
vec_max(const __m128& a, const __m128& b)
{
  __m128 r;

  r.xfrd[0] = a.xfrd[0] > b.xfrd[0] ? a.xfrd[0] : b.xfrd[0];
  r.xfrd[1] = a.xfrd[1] > b.xfrd[1] ? a.xfrd[1] : b.xfrd[1];
  r.xfrd[2] = a.xfrd[2] > b.xfrd[2] ? a.xfrd[2] : b.xfrd[2];
  r.xfrd[3] = a.xfrd[3] > b.xfrd[3] ? a.xfrd[3] : b.xfrd[3];

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_max(const __m128d& a, const __m128d& b)
{
  __m128d r;

  r.xdrd[0] = a.xdrd[0] > b.xdrd[0] ? a.xdrd[0] : b.xdrd[0];
  r.xdrd[1] = a.xdrd[1] > b.xdrd[1] ? a.xdrd[1] : b.xdrd[1];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_max(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if ((a.type == b.type) && (a.type == __s128i::SignedInt)) {
    r.xird[0] = a.xird[0] > b.xird[0] ? a.xird[0] : b.xird[0];
    r.xird[1] = a.xird[1] > b.xird[1] ? a.xird[1] : b.xird[1];
    r.xird[2] = a.xird[2] > b.xird[2] ? a.xird[2] : b.xird[2];
    r.xird[3] = a.xird[3] > b.xird[3] ? a.xird[3] : b.xird[3];
  } else if ((a.type == b.type) && (a.type == __s128i::UnsignedInt)) {
    r.xui[0] = a.xui[0] > b.xui[0] ? a.xui[0] : b.xui[0];
    r.xui[1] = a.xui[1] > b.xui[1] ? a.xui[1] : b.xui[1];
    r.xui[2] = a.xui[2] > b.xui[2] ? a.xui[2] : b.xui[2];
    r.xui[3] = a.xui[3] > b.xui[3] ? a.xui[3] : b.xui[3];
  } else if ((a.type == b.type) && (a.type == __s128i::SignedLong)) {
    r.xlrd[0] = a.xlrd[0] > b.xlrd[0] ? a.xlrd[0] : b.xlrd[0];
    r.xlrd[1] = a.xlrd[1] > b.xlrd[1] ? a.xlrd[1] : b.xlrd[1];
  } else if ((a.type == b.type) && (a.type == __s128i::UnsignedLong)) {
    r.xul[0] = a.xul[0] > b.xul[0] ? a.xul[0] : b.xul[0];
    r.xul[1] = a.xul[1] > b.xul[1] ? a.xul[1] : b.xul[1];
  }

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_min(const __m128& a, const __m128& b)
{
  __m128 r;

  r.xfrd[0] = a.xfrd[0] < b.xfrd[0] ? a.xfrd[0] : b.xfrd[0];
  r.xfrd[1] = a.xfrd[1] < b.xfrd[1] ? a.xfrd[1] : b.xfrd[1];
  r.xfrd[2] = a.xfrd[2] < b.xfrd[2] ? a.xfrd[2] : b.xfrd[2];
  r.xfrd[3] = a.xfrd[3] < b.xfrd[3] ? a.xfrd[3] : b.xfrd[3];

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_min(const __m128d& a, const __m128d& b)
{
  __m128d r;

  r.xdrd[0] = a.xdrd[0] < b.xdrd[0] ? a.xdrd[0] : b.xdrd[0];
  r.xdrd[1] = a.xdrd[1] < b.xdrd[1] ? a.xdrd[1] : b.xdrd[1];

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_min(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if ((a.type == b.type) && (a.type == __s128i::SignedInt)) {
    r.xird[0] = a.xird[0] < b.xird[0] ? a.xird[0] : b.xird[0];
    r.xird[1] = a.xird[1] < b.xird[1] ? a.xird[1] : b.xird[1];
    r.xird[2] = a.xird[2] < b.xird[2] ? a.xird[2] : b.xird[2];
    r.xird[3] = a.xird[3] < b.xird[3] ? a.xird[3] : b.xird[3];
  } else if ((a.type == b.type) && (a.type == __s128i::UnsignedInt)) {
    r.xui[0] = a.xui[0] < b.xui[0] ? a.xui[0] : b.xui[0];
    r.xui[1] = a.xui[1] < b.xui[1] ? a.xui[1] : b.xui[1];
    r.xui[2] = a.xui[2] < b.xui[2] ? a.xui[2] : b.xui[2];
    r.xui[3] = a.xui[3] < b.xui[3] ? a.xui[3] : b.xui[3];
  } else if ((a.type == b.type) && (a.type == __s128i::SignedLong)) {
    r.xlrd[0] = a.xlrd[0] < b.xlrd[0] ? a.xlrd[0] : b.xlrd[0];
    r.xlrd[1] = a.xlrd[1] < b.xlrd[1] ? a.xlrd[1] : b.xlrd[1];
  } else if ((a.type == b.type) && (a.type == __s128i::UnsignedLong)) {
    r.xul[0] = a.xul[0] < b.xul[0] ? a.xul[0] : b.xul[0];
    r.xul[1] = a.xul[1] < b.xul[1] ? a.xul[1] : b.xul[1];
  }

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_cmplt(const __m128i& a, const __m128i& b)
{
  __m128i r;

  if (a.xird[0] < b.xird[0])
    r.xird[0] = (unsigned) 0xFFFFFFFF;

  if (a.xird[1] < b.xird[1])
    r.xird[1] = (unsigned) 0xFFFFFFFF;

  if (a.xird[2] < b.xird[2])
    r.xird[2] = (unsigned) 0xFFFFFFFF;

  if (a.xird[3] < b.xird[3])
    r.xird[3] = (unsigned) 0xFFFFFFFF;

  return r;
}

static inline __m128
__attribute__((always_inline))
vec_sel(const __m128& a, const __m128& b, const __m128& c)
{
  __m128 r;

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 32; ++j) {
      if (c.xi[i] & (1 << j)) {
        if (b.xi[i] & (1 << j))
          r.xi[i] |= (b.xi[i] & (1 << j));
        else
          r.xi[i] &= ~(b.xi[i] & (1 << j));
      } else {
        if (a.xi[i] & (1 << j))
          r.xi[i] |= (a.xi[i] & (1 << j));
        else
          r.xi[i] &= ~(a.xi[i] & (1 << j));
      }
    }
  }

  return r;
}

static inline __m128d
__attribute__((always_inline))
vec_sel(const __m128d& a, const __m128d& b, const __m128d& c)
{
  __m128d r;

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 32; ++j) {
      if (c.xi[i] & (1 << j)) {
        if (b.xi[i] & (1 << j))
          r.xi[i] |= (b.xi[i] & (1 << j));
        else
          r.xi[i] &= ~(b.xi[i] & (1 << j));
      } else {
        if (a.xi[i] & (1 << j))
          r.xi[i] |= (a.xi[i] & (1 << j));
        else
          r.xi[i] &= ~(a.xi[i] & (1 << j));
      }
    }
  }

  return r;
}

static inline __m128i
__attribute__((always_inline))
vec_sel(const __m128i& a, const __m128i& b, const __m128i& c)
{
  __m128i r;

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 32; ++j) {
      if (c.xi[i] & (1 << j)) {
        if (b.xi[i] & (1 << j))
          r.xi[i] |= (b.xi[i] & (1 << j));
        else
          r.xi[i] &= ~(b.xi[i] & (1 << j));
      } else {
        if (a.xi[i] & (1 << j))
          r.xi[i] |= (a.xi[i] & (1 << j));
        else
          r.xi[i] &= ~(a.xi[i] & (1 << j));
      }
    }
  }

  return r;
}

static inline void
__attribute__((always_inline))
vec_sti(const __m128i& vi, int v, unsigned char* t)
{
  __m128i vix = v ? vi + v : vi;
  unsigned char* x = vix.c;

  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  t[12] = x[12];
  t[13] = x[13];
  t[14] = x[14];
  t[15] = x[15];
}

static inline void
__attribute__((always_inline))
vec_stf(const __m128& vf, int v, unsigned char* t)
{
  __m128 vfx = v ? vf + v : vf;
  unsigned char* x = vfx.c;

  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  t[12] = x[12];
  t[13] = x[13];
  t[14] = x[14];
  t[15] = x[15];
}

static inline void
__attribute__((always_inline))
vec_std(const __m128d& vd, int v, unsigned char* t)
{
  __m128d vdx = v ? vd + v : vd;
  unsigned char* x = vdx.c;

  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  t[4] = x[4];
  t[5] = x[5];
  t[6] = x[6];
  t[7] = x[7];
  t[8] = x[8];
  t[9] = x[9];
  t[10] = x[10];
  t[11] = x[11];
  t[12] = x[12];
  t[13] = x[13];
  t[14] = x[14];
  t[15] = x[15];
}

/*
 * No corresponding Altivec intrinsic to generate a scalar mask
 * from corresponding vector elements.
 */
static inline unsigned int
__attribute__((always_inline))
_mm_movemask_epi8(const __m128i& a)
{
  unsigned char t[16] __attribute__((aligned(16)));
  unsigned int r;
  int i;

  vec_st(a, 0, t);
  r = 0;
  for (i = 0; i < 16; i++) {
    r = (r << 1) | (t[i] >> 7);
  }

  return r;
}

static inline unsigned int
__attribute__((always_inline))
_mm_movemask_epi32(const __m128i& a)
{
  unsigned int t[4] __attribute__((aligned(16)));
  unsigned int r;
  int i;

  vec_st(a, 0, t);
  r = 0;
  for (i = 0; i < 4; i++) {
    r = (r << 1) | (t[i] >> 31);
  }

  return r;
}

static inline unsigned int
__attribute__((always_inline))
_mm_movemask_ps(const __m128& a)
{
  return ((a.xui[3] >> 31) << 3) | ((a.xui[2] >> 31) << 2) |
         ((a.xui[1] >> 31) << 1) | ((a.xui[0] >> 31) << 0);
}

static inline unsigned int
__attribute__((always_inline))
_mm_movemask_pd(const __m128d& a)
{
  return ((a.xul[1] >> 63) << 1) | ((a.xul[0] >> 63) << 0);
}

static inline __m128i
__attribute__((always_inline))
_mm_blend_epi32(const __m128i& a, const __m128i& b, int imm8)
{
  unsigned int t[4] __attribute__((aligned(16)));
  int i;

  vec_st(a, 0, t);
  for (i = 0; i < 3; i++) {
    if (imm8 & 0x1)
      t[i] = vec_extract(b, i);
    imm8 >>= 1;
  }

  // FIXME: Check the cast below.
  return vec_ld(0, (__m128i*) t);
}

static inline __m128
__attribute__((always_inline))
_mm_setr_ps(float e3, float e2, float e1, float e0)
{
  __m128 e = { e3, e2, e1, e0 };
  return e;
}

static inline __m128d
__attribute__((always_inline))
_mm_setr_pd(double e1, double e0)
{
  __m128d e = { e1, e0 };
  return e;
}

static inline __m128d
__attribute__((always_inline))
_mm_shuffle_pd(const __m128d& a, const __m128d& b, int imm8)
{
  double r[2];
  r[0] = imm8 & 0x1 ? vec_extract(a, 1) : vec_extract(a, 0);
  r[1] = imm8 & 0x2 ? vec_extract(b, 1) : vec_extract(b, 0);

  return vec_ld(0, (__m128d *)r);
}

/*
 * Quick way to determine whether any element in a vector mask
 * register is set.
 *
 * No corresponding Altivec intrinsic.
 */
static inline unsigned int
__attribute__((always_inline))
_vec_any_nz(const __m128i& a)
{
  return vec_any_ne(a, (__typeof__(a)) vec_splats(0));
}

static inline __m128d
__attribute__((always_inline))
_mm_cvtepi32_pd(const __m128i& a)
{
  __m128d r;

  r = vec_insert(1.0 * vec_extract(a, 0), r, 0);
  r = vec_insert(1.0 * vec_extract(a, 2), r, 1);

  return r;
}

static inline __m128d
__attribute__((always_inline))
_mm_min_sd(const __m128d& a, const __m128d& b)
{
  double aa = vec_extract(a, 0);
  double bb = vec_extract(b, 0);
  aa = aa < bb ? aa : bb;
  return vec_insert(aa, a, 0);
}

static inline __m128d
__attribute__((always_inline))
_mm_max_sd(const __m128d& a, const __m128d& b)
{
  double aa = vec_extract(a, 0);
  double bb = vec_extract(b, 0);
  aa = aa > bb ? aa : bb;
  return vec_insert(aa, a, 0);
}


/*
 * Logical
 */

#define	_mm_andnot_ps(_v,_w) vec_andc(_w,_v)     // different oder of arguments
#define	_mm_andnot_pd(_v,_w) vec_andc(_w,_v)     // different oder of arguments
#define	_mm_and_ps(_v,_w) vec_and(_v,_w)
#define	_mm_and_pd(_v,_w) vec_and(_v,_w)
#define	_mm_and_si128(_v,_w) vec_and(_v,_w)
#define	_mm_andnot_si128(_v,_w) vec_andc(_w,_v)  // different order of arguments
#define	_mm_or_ps(_v,_w) vec_or(_v,_w)
#define	_mm_or_pd(_v,_w) vec_or(_v,_w)
#define	_mm_or_si128(_v,_w) vec_or(_v,_w)
#define	_mm_xor_ps(_v,_w) vec_xor(_v,_w)
#define	_mm_xor_pd(_v,_w) vec_xor(_v,_w)
#define	_mm_xor_si128(_v,_w) vec_xor(_v,_w)

/*
 * Broadcast
 */

#define	_mm_set1_epi32(_v) (__m128i)vec_splats((int)_v)
#define	_mm_set1_epi64x(_v) (__m128i)vec_splats((long int)_v)
#define	_mm_set1_ps(_v) (__m128)vec_splats((float)_v)
#define	_mm_set1_pd(_v) (__m128d)vec_splats((double)_v)
//#define	_mm_setr_ps(_e,_f) (__m128d)vec_insert(_e, (__m128d)vec_splats(_f), 0)
//#define	_mm_setr_pd(_e,_f) (__m128d)vec_insert(_e, (__m128d)vec_splats(_f), 0)
#define	_mm_setzero_ps() (__m128)vec_splats((float)0.0)
#define	_mm_setzero_pd() (__m128d)vec_splats((double)0.0)

#define	_mm_cvtps_epi32(_v) vec_cts(_v,0)
// Need inline version #define	_mm_cvtepi32_pd(_v) vec_ctd(_v,0)
#define	_mm_cvtepi32_ps(_v) vec_ctf(_v,0)
#define	_mm_cvtss_f32(_v) (float)vec_extract(_v,0)
#define	_mm_cvtsd_f64(_v) (double)vec_extract(_v,0)
//#define	_mm_cvtpd_ps(_v) (__m128)vec_cvf(_v)	// Does not work
#define	_mm_cvtpd_ps(_v) vec_insert((float)vec_extract(_v,1), (vec_insert((float)vec_extract(_v,0), (__m128)vec_splats((float)0.0), 0)), 1)
#define	_mm_cvtss_sd(_v,_w) vec_insert((double) vec_extract(_w, 0), _v, 0)
#define _mm_extract_ps(_v,_i) vec_extract(_v,_i)

/*
 * Floating point
 */

#define	_mm_add_ps(_v,_w) vec_add(_v,_w)
#define	_mm_add_pd(_v,_w) vec_add(_v,_w)
#define	_mm_add_epi64(_v,_w) vec_add(_v,_w)
#define	_mm_mul_ps(_v,_w) vec_mul(_v,_w)
#define	_mm_mul_pd(_v,_w) vec_mul(_v,_w)
#define	_mm_sub_ps(_v,_w) vec_sub(_v,_w)
#define	_mm_sub_pd(_v,_w) vec_sub(_v,_w)
#define	_mm_sub_epi32(_v,_w) vec_sub(_v,_w)
#define	_mm_sub_epi64(_v,_w) vec_sub(_v,_w)
#define	_mm_div_ps(_v,_w) vec_div(_v,_w)
#define	_mm_div_pd(_v,_w) vec_div(_v,_w)
#define	_mm_sqrt_ps(_v) vec_sqrt(_v)
#define	_mm_sqrt_pd(_v) vec_sqrt(_v)

#define	_mm_add_ss(_s,_t) (_s+_t)
#define	_mm_add_sd(_s,_t) (_s+_t)
#define	_mm_mul_ss(_s,_t) (_s*_t)
#define	_mm_mul_sd(_s,_t) (_s*_t)
#define	_mm_sub_ss(_s,_t) (_s-_t)
#define	_mm_sub_sd(_s,_t) (_s-_t)
#define	_mm_div_ss(_s,_t) (_s/_t)
#define	_mm_div_sd(_s,_t) (_s/_t)

#define	_mm_floor_ps(_v) vec_floor(_v)
#define	_mm_floor_pd(_v) vec_floor(_v)

/*
 * FMA instructions.
 *
 * _mm_fnmadd_p{s,d} not the same as Altivec intrinsic vec_nmadd(a,b,c).
 * Altivec returns: -(a*b+c).
 * We want: (-(a*b)+c)
 */

#define	_mm_fmadd_ps(_v,_w,_x) vec_madd(_v,_w,_x)
#define	_mm_fmadd_pd(_v,_w,_x) vec_madd(_v,_w,_x)
#define	_mm_fmsub_ps(_v,_w,_x) vec_msub(_v,_w,_x)
#define	_mm_fmsub_pd(_v,_w,_x) vec_msub(_v,_w,_x)
#define	_mm_fnmadd_ps(_v,_w,_x) vec_madd((-(_v)),_w,_x)
#define	_mm_fnmadd_pd(_v,_w,_x) vec_madd((-(_v)),_w,_x)
#define	_mm_min_epi32(_v,_w) vec_min(_v,_w)
#define	_mm_max_epi32(_v,_w) vec_max(_v,_w)
#define	_mm_max_epu32(_v,_w) vec_max(_v,_w)
//#define	_mm_min_sd(_v,_w)

#define	_mm_fmadd_ss(_v,_w,_x) vec_madd(_v,_w,_x)//fmaf(_v,_w,_x) //((_v*_w)+_x)
#define	_mm_fmadd_sd(_v,_w,_x) vec_madd(_v,_w,_x)//fmaf(_v,_w,_x) //((_v*_w)+_x)
#define	_mm_fmsub_ss(_v,_w,_x) vec_msub(_v,_w,_x)//fmsf(_v,_w,_x) //((_v*_w)-_x)
#define	_mm_fmsub_sd(_v,_w,_x) vec_msub(_v,_w,_x)//fmsf(_v,_w,_x) //((_v*_w)-_x)

/*
 * Integer.
 */

#define	_mm_add_epi32(_v,_w) vec_add(_v,_w)
#define	_mm_sub_epi32(_v,_w) vec_sub(_v,_w)

/*
 * Merge.
 */

#define	_mm_blendv_ps(_v,_w,_m) vec_sel(_v,_w,_m)
#define	_mm_blendv_pd(_v,_w,_m) vec_sel(_v,_w,_m)

/*
 * Miscelaneous:
 * Vector op constant
 * Casting
 */

#define	_mm_castps_si128(_v) (__m128i)(_v)
#define	_mm_castpd_si128(_v) (__m128i)(_v)
#define	_mm_slli_epi32(_v,_c) vec_sl(_v,vec_splats((unsigned int)_c))
#define	_mm_slli_epi64(_v,_c) (__m128i)vec_sl(_v,vec_splats((unsigned long)_c))
#define	_mm_sllv_epi64(_v,_w) vec_sl((__m128i)_v,_w)
#define	_mm_srli_epi32(_v,_c) vec_sr(_v,vec_splats((unsigned int)_c))
#define	_mm_srli_epi64(_v,_c) vec_sr(_v,vec_splats((unsigned long)_c))

/*
 * Comparision.
 *
 * The following 4 macros stole shamelessly from:
 * https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
 */

#define	_CAT(_a,_b,...) _a##_b
#define	_EMPTY()
#define	_DEFER(id) id _EMPTY()
#define	_EXPAND1(...) __VA_ARGS__
#define	_EXPAND(...) _EXPAND1(_EXPAND1(__VA_ARGS__))

#define	__CMP_EQ_OQ(_v,_w) (__typeof__(_v))vec_cmpeq(_v,_w)
#define	__CMP_EQ_OS(_v,_w) (__typeof__(_v))vec_cmpeq(_v,_w)
#define	__CMP_LE_OQ(_v,_w) (__typeof__(_v))vec_cmple(_v,_w)
#define	__CMP_LT_OS(_v,_w) (__typeof__(_v))vec_cmplt(_v,_w)
#define	__CMP_LT_OQ(_v,_w) (__typeof__(_v))vec_cmplt(_v,_w)
#define	__CMP_GE_OS(_v,_w) (__typeof__(_v))vec_cmpge(_v,_w)
#define	__CMP_GT_OS(_v,_w) (__typeof__(_v))vec_cmpgt(_v,_w)
#define	__CMP_GT_OQ(_v,_w) (__typeof__(_v))vec_cmpgt(_v,_w)
//#define	__CMP_NEQ_UQ(_v,_w) (typeof(_v))vec_andc((__m128i)vec_splats(0xffffffff),(__m128i)vec_cmpeq(_v, _w))
#define	__CMP_NEQ_UQ(_v,_w) \
  (__typeof__(_v))vec_andc((__m128i)vec_splats(-1),(__m128i)vec_cmpeq(_v, _w))
#define	__CMP_NLT_UQ(_v,_w) \
  (__typeof__(_v))vec_andc((__m128i)vec_splats(-1),(__m128i)vec_cmplt(_v, _w))
#define	__CMP_NGE_UQ(_v,_w) \
  (__typeof__(_v))vec_andc((__m128i)vec_splats(-1),(__m128i)vec_cmpge(_v, _w))

#define	_mm_cmpeq_epi32(_v,_w) (__m128i)vec_cmpeq(_v,_w)
#define	_mm_cmpeq_epi64(_v,_w) (__m128i)vec_cmpeq(_v,_w)
#define	_mm_cmpgt_epi32(_v,_w) (__m128i)vec_cmpgt(_v,_w)
#define	_mm_cmpgt_epi64(_v,_w) (__m128i)vec_cmpgt(_v,_w)
#define	_mm_cmple_ps(_v,_w) (__m128i)vec_cmple(_v,_w)
#define	_mm_cmplt_ps(_v,_w) (__m128i)vec_cmplt(_v,_w)
#define	_mm_cmpeq_ps(_v,_w) (__m128i)vec_cmpeq(_v,_w)
#define	_mm_cmp_ps(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))
#define	_mm_cmp_pd(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))
#define	_mm_cmp_ss(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))
#define	_mm_cmp_sd(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))

/*
 * More macros that have to have secondary expansion.
 */

#define __MM_FROUND_TO_ZERO(_v) vec_trunc(_v)
// - does seem to exist with GCC 5.4 #define __MM_FROUND_TO_ZERO(_v) vec_roundz(_v)
#define	_mm_round_ps(_v,_m) _EXPAND(_DEFER(_CAT(_,_m))(_v))
#define	_mm_round_pd(_v,_m) _EXPAND(_DEFER(_CAT(_,_m))(_v))
#endif


#ifdef	DEBUG
#include <stdio.h>
static inline void
__attribute__((always_inline))
_dumpfvec(__m128 a, char *t)
{
  int i;
  printf("%s:", t);
  for (i = 0 ; i < 4 ; i++) {
    printf(" %#x", *(unsigned int *)&a[i]);
  }
  printf("\n");
}
static inline void
__attribute__((always_inline))
_dumpdvec(__m128d a, char *t)
{
  int i;
  printf("%s:", t);
  for (i = 0 ; i < 2 ; i++) {
    printf(" %#lx", *(unsigned long int *)&a[i]);
  }
  printf("\n");
}

#endif
