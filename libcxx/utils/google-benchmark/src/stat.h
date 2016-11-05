#ifndef BENCHMARK_STAT_H_
#define BENCHMARK_STAT_H_

#include <cmath>
#include <limits>
#include <ostream>
#include <type_traits>

namespace benchmark {

template <typename VType, typename NumType>
class Stat1;

template <typename VType, typename NumType>
class Stat1MinMax;

typedef Stat1<float, int64_t> Stat1_f;
typedef Stat1<double, int64_t> Stat1_d;
typedef Stat1MinMax<float, int64_t> Stat1MinMax_f;
typedef Stat1MinMax<double, int64_t> Stat1MinMax_d;

template <typename VType>
class Vector2;
template <typename VType>
class Vector3;
template <typename VType>
class Vector4;

template <typename VType, typename NumType>
class Stat1 {
 public:
  typedef Stat1<VType, NumType> Self;

  Stat1() { Clear(); }
  // Create a sample of value dat and weight 1
  explicit Stat1(const VType &dat) {
    sum_ = dat;
    sum_squares_ = Sqr(dat);
    numsamples_ = 1;
  }
  // Create statistics for all the samples between begin (included)
  // and end(excluded)
  explicit Stat1(const VType *begin, const VType *end) {
    Clear();
    for (const VType *item = begin; item < end; ++item) {
      (*this) += Stat1(*item);
    }
  }
  // Create a sample of value dat and weight w
  Stat1(const VType &dat, const NumType &w) {
    sum_ = w * dat;
    sum_squares_ = w * Sqr(dat);
    numsamples_ = w;
  }
  // Copy operator
  Stat1(const Self &stat) {
    sum_ = stat.sum_;
    sum_squares_ = stat.sum_squares_;
    numsamples_ = stat.numsamples_;
  }

  void Clear() {
    numsamples_ = NumType();
    sum_squares_ = sum_ = VType();
  }

  Self &operator=(const Self &stat) {
    sum_ = stat.sum_;
    sum_squares_ = stat.sum_squares_;
    numsamples_ = stat.numsamples_;
    return (*this);
  }
  // Merge statistics from two sample sets.
  Self &operator+=(const Self &stat) {
    sum_ += stat.sum_;
    sum_squares_ += stat.sum_squares_;
    numsamples_ += stat.numsamples_;
    return (*this);
  }
  // The operation opposite to +=
  Self &operator-=(const Self &stat) {
    sum_ -= stat.sum_;
    sum_squares_ -= stat.sum_squares_;
    numsamples_ -= stat.numsamples_;
    return (*this);
  }
  // Multiply the weight of the set of samples by a factor k
  Self &operator*=(const VType &k) {
    sum_ *= k;
    sum_squares_ *= k;
    numsamples_ *= k;
    return (*this);
  }

  // Merge statistics from two sample sets.
  Self operator+(const Self &stat) const { return Self(*this) += stat; }

  // The operation opposite to +
  Self operator-(const Self &stat) const { return Self(*this) -= stat; }

  // Multiply the weight of the set of samples by a factor k
  Self operator*(const VType &k) const { return Self(*this) *= k; }

  // Return the total weight of this sample set
  NumType numSamples() const { return numsamples_; }

  // Return the sum of this sample set
  VType Sum() const { return sum_; }

  // Return the mean of this sample set
  VType Mean() const {
    if (numsamples_ == 0) return VType();
    return sum_ * (1.0 / numsamples_);
  }

  // Return the mean of this sample set and compute the standard deviation at
  // the same time.
  VType Mean(VType *stddev) const {
    if (numsamples_ == 0) return VType();
    VType mean = sum_ * (1.0 / numsamples_);
    if (stddev) {
      VType avg_squares = sum_squares_ * (1.0 / numsamples_);
      *stddev = Sqrt(avg_squares - Sqr(mean));
    }
    return mean;
  }

  // Return the standard deviation of the sample set
  VType StdDev() const {
    if (numsamples_ == 0) return VType();
    VType mean = Mean();
    VType avg_squares = sum_squares_ * (1.0 / numsamples_);
    return Sqrt(avg_squares - Sqr(mean));
  }

 private:
  static_assert(std::is_integral<NumType>::value &&
                    !std::is_same<NumType, bool>::value,
                "NumType must be an integral type that is not bool.");
  // Let i be the index of the samples provided (using +=)
  // and weight[i],value[i] be the data of sample #i
  // then the variables have the following meaning:
  NumType numsamples_;  // sum of weight[i];
  VType sum_;           // sum of weight[i]*value[i];
  VType sum_squares_;   // sum of weight[i]*value[i]^2;

  // Template function used to square a number.
  // For a vector we square all components
  template <typename SType>
  static inline SType Sqr(const SType &dat) {
    return dat * dat;
  }

  template <typename SType>
  static inline Vector2<SType> Sqr(const Vector2<SType> &dat) {
    return dat.MulComponents(dat);
  }

  template <typename SType>
  static inline Vector3<SType> Sqr(const Vector3<SType> &dat) {
    return dat.MulComponents(dat);
  }

  template <typename SType>
  static inline Vector4<SType> Sqr(const Vector4<SType> &dat) {
    return dat.MulComponents(dat);
  }

  // Template function used to take the square root of a number.
  // For a vector we square all components
  template <typename SType>
  static inline SType Sqrt(const SType &dat) {
    // Avoid NaN due to imprecision in the calculations
    if (dat < 0) return 0;
    return sqrt(dat);
  }

  template <typename SType>
  static inline Vector2<SType> Sqrt(const Vector2<SType> &dat) {
    // Avoid NaN due to imprecision in the calculations
    return Max(dat, Vector2<SType>()).Sqrt();
  }

  template <typename SType>
  static inline Vector3<SType> Sqrt(const Vector3<SType> &dat) {
    // Avoid NaN due to imprecision in the calculations
    return Max(dat, Vector3<SType>()).Sqrt();
  }

  template <typename SType>
  static inline Vector4<SType> Sqrt(const Vector4<SType> &dat) {
    // Avoid NaN due to imprecision in the calculations
    return Max(dat, Vector4<SType>()).Sqrt();
  }
};

// Useful printing function
template <typename VType, typename NumType>
std::ostream &operator<<(std::ostream &out, const Stat1<VType, NumType> &s) {
  out << "{ avg = " << s.Mean() << " std = " << s.StdDev()
      << " nsamples = " << s.NumSamples() << "}";
  return out;
}

// Stat1MinMax: same as Stat1, but it also
// keeps the Min and Max values; the "-"
// operator is disabled because it cannot be implemented
// efficiently
template <typename VType, typename NumType>
class Stat1MinMax : public Stat1<VType, NumType> {
 public:
  typedef Stat1MinMax<VType, NumType> Self;

  Stat1MinMax() { Clear(); }
  // Create a sample of value dat and weight 1
  explicit Stat1MinMax(const VType &dat) : Stat1<VType, NumType>(dat) {
    max_ = dat;
    min_ = dat;
  }
  // Create statistics for all the samples between begin (included)
  // and end(excluded)
  explicit Stat1MinMax(const VType *begin, const VType *end) {
    Clear();
    for (const VType *item = begin; item < end; ++item) {
      (*this) += Stat1MinMax(*item);
    }
  }
  // Create a sample of value dat and weight w
  Stat1MinMax(const VType &dat, const NumType &w)
      : Stat1<VType, NumType>(dat, w) {
    max_ = dat;
    min_ = dat;
  }
  // Copy operator
  Stat1MinMax(const Self &stat) : Stat1<VType, NumType>(stat) {
    max_ = stat.max_;
    min_ = stat.min_;
  }

  void Clear() {
    Stat1<VType, NumType>::Clear();
    if (std::numeric_limits<VType>::has_infinity) {
      min_ = std::numeric_limits<VType>::infinity();
      max_ = -std::numeric_limits<VType>::infinity();
    } else {
      min_ = std::numeric_limits<VType>::max();
      max_ = std::numeric_limits<VType>::min();
    }
  }

  Self &operator=(const Self &stat) {
    this->Stat1<VType, NumType>::operator=(stat);
    max_ = stat.max_;
    min_ = stat.min_;
    return (*this);
  }
  // Merge statistics from two sample sets.
  Self &operator+=(const Self &stat) {
    this->Stat1<VType, NumType>::operator+=(stat);
    if (stat.max_ > max_) max_ = stat.max_;
    if (stat.min_ < min_) min_ = stat.min_;
    return (*this);
  }
  // Multiply the weight of the set of samples by a factor k
  Self &operator*=(const VType &stat) {
    this->Stat1<VType, NumType>::operator*=(stat);
    return (*this);
  }
  // Merge statistics from two sample sets.
  Self operator+(const Self &stat) const { return Self(*this) += stat; }
  // Multiply the weight of the set of samples by a factor k
  Self operator*(const VType &k) const { return Self(*this) *= k; }

  // Return the maximal value in this sample set
  VType Max() const { return max_; }
  // Return the minimal value in this sample set
  VType Min() const { return min_; }

 private:
  // The - operation makes no sense with Min/Max
  // unless we keep the full list of values (but we don't)
  // make it private, and let it undefined so nobody can call it
  Self &operator-=(const Self &stat);  // senseless. let it undefined.

  // The operation opposite to -
  Self operator-(const Self &stat) const;  // senseless. let it undefined.

  // Let i be the index of the samples provided (using +=)
  // and weight[i],value[i] be the data of sample #i
  // then the variables have the following meaning:
  VType max_;  // max of value[i]
  VType min_;  // min of value[i]
};

// Useful printing function
template <typename VType, typename NumType>
std::ostream &operator<<(std::ostream &out,
                         const Stat1MinMax<VType, NumType> &s) {
  out << "{ avg = " << s.Mean() << " std = " << s.StdDev()
      << " nsamples = " << s.NumSamples() << " min = " << s.Min()
      << " max = " << s.Max() << "}";
  return out;
}
}  // end namespace benchmark

#endif  // BENCHMARK_STAT_H_
